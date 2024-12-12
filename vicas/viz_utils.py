import copy
import numpy as np
import os
import os.path as osp
import multiprocessing as mp
import pycocotools.mask as mt

from typing import List, Dict, Optional, Tuple, Any
from tqdm import tqdm
from glob import glob
from vicas.utils import frames_to_video
from vicas.caption_parsing import parse_caption

try:
    import cv2
    CV2_IMPORTED = True
except ImportError as _:
    CV2_IMPORTED = False


BORDER_SIZE = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 1
MAX_LINES_EN = 4


def create_color_map(N=256, normalized=False):
    # Copied from the DAVIS dataset API: https://github.com/davisvideochallenge/davis2017-evaluation/blob/master/davis2017/utils.py
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def overlay_mask_on_image(image, mask, mask_opacity=0.6, mask_color=(0, 255, 0), border_thickness=0):
    if mask.ndim == 3:
        assert mask.shape[2] == 1
        _mask = mask.squeeze(axis=2)
    else:
        _mask = mask

    mask_bgr = np.stack((_mask, _mask, _mask), axis=2)
    masked_image = np.where(mask_bgr > 0, mask_color, image)
    overlayed_image = ((mask_opacity * masked_image) + ((1. - mask_opacity) * image)).astype(np.uint8)

    if border_thickness > 0:
        _mask = _mask.astype(np.uint8)
        assert border_thickness % 2 == 1  # odd number
        kernel = np.ones((border_thickness, border_thickness), np.uint8)
        edge_mask = cv2.dilate(_mask, kernel, iterations=1) - _mask
        edge_mask = np.stack([edge_mask, edge_mask, edge_mask], axis=2)
        mask_color = np.array(mask_color, np.uint8)[None, None, :]
        mask_color = np.repeat(mask_color, image.shape[0], 0)
        mask_color = np.repeat(mask_color, image.shape[1], 1)
        overlayed_image = np.where(edge_mask > 0, mask_color, overlayed_image)

    return overlayed_image 


def _handle_cv2umat(img):
    if isinstance(img, cv2.UMat):
        return img.get()
    else:
        return img


def annotate_image_instance(image, mask=None, color=None, **kwargs):
    """
    :param image: np.ndarray(H, W, 3)
    :param mask: np.ndarray(H, W)
    :param color: tuple/list(int, int, int) in range [0, 255]
    :param label: str
    :param kwargs: "bbox_thickness", "text_font", "font_size", "mask_opacity", "text_placement"
    :return: np.ndarray(H, W, 3)
    """
    # parse color
    color = tuple(color)  # cv2 does not like colors as lists

    assert image.shape[:2] == mask.shape, f"Shape mismatch between image {image.shape} and mask {mask.shape}"
    annotated_image = overlay_mask_on_image(
        image, mask, mask_color=color, mask_opacity=kwargs.get("mask_opacity", 0.6),
        border_thickness=kwargs.get("mask_border", 0)
    )

    # sometimes OpenCV will retun an object of type cv2.UMat instead of a numpy array (don't know why)
    return _handle_cv2umat(annotated_image)


def rle_to_mask(rle_dict) -> np.ndarray:
    d = copy.deepcopy(rle_dict)
    d['counts'] = d['counts'].encode("utf-8")
    return mt.decode(d).astype(np.uint8)


def render_char(img: np.ndarray, char: str, color: List[int], orig_x, orig_y, font, font_scale, font_thickness):
    assert len(char) == 1
    assert len(color) == 3  # BGR
    text_w, text_h = cv2.getTextSize(char, font, font_scale, thickness=font_thickness)[0]
    color = tuple(color)
    # font_scale = FONT_SCALE if color == (0, 0, 0) else font_scale + 0.2
    img = cv2.putText(img, char, (orig_x, orig_y), font, fontScale=font_scale, color=color, thickness=font_thickness, lineType=cv2.LINE_AA)
    if color != (0, 0, 0):
        offset = 7
        thickness = 2
        img = cv2.rectangle(img, (orig_x, orig_y+offset-thickness), (orig_x+text_w, orig_y+offset), color=color, thickness=-1)
    return img, text_w


def get_text_size(txt, font, font_scale, font_thickness):
    # calculates character by character which is different from applying it to the whole string
    _, text_h = cv2.getTextSize(txt, font, font_scale, thickness=font_thickness)[0]
    text_w = 0
    for ch in txt:
        ch_w, _ = cv2.getTextSize(ch, font, font_scale, thickness=font_thickness)[0]
        text_w += ch_w

    return text_w, text_h


def split_caption_into_lines(caption: str, img_width, font, font_scale, font_thickness):
    tokens = caption.split(" ")
    lines = []
    curr_line = ""
    for t in tokens:
        new_line = curr_line + t + " "
        # text_w, _ = cv2.getTextSize(new_line, FONT, FONT_SCALE, thickness=FONT_THICKNESS)[0]
        text_w, _ = get_text_size(new_line, font=font, font_scale=font_scale, font_thickness=font_thickness)
        if text_w > img_width:
            lines.append(curr_line)
            curr_line = t + " "
        else:
            curr_line = new_line

    if curr_line != "":
        lines.append(curr_line)
    return lines


def apply_caption_to_img(img: np.ndarray, caption: str, colors: List[List[int]], font=FONT, font_scale=FONT_SCALE, font_thickness=FONT_THICKNESS):
    assert len(colors) == len(caption)
    img = cv2.copyMakeBorder(img, 0, BORDER_SIZE, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    height, width = img.shape[:2]
    # _, text_h = cv2.getTextSize(caption, FONT, FONT_SCALE, thickness=FONT_THICKNESS)[0]
    _, text_h = get_text_size(caption, font=font, font_scale=font_scale, font_thickness=font_thickness)

    lines = split_caption_into_lines(caption, width, font=font, font_scale=font_scale, font_thickness=font_thickness)

    color_idx = 0
    line_blocks = []
    for line in lines:
        block = np.ones((text_h+20, width, 3), np.uint8) * 255
        # line_w = cv2.getTextSize(line, FONT, FONT_SCALE, thickness=FONT_THICKNESS)[0][0]
        line_w, _ = get_text_size(line, font=font, font_scale=font_scale, font_thickness=font_thickness)
        line_colors = colors[color_idx:color_idx+len(line)]
        origin_x, origin_y = (width // 2) - (line_w // 2), text_h+10
        
        for ch, color in zip(line, line_colors):
            block, width_ch = render_char(block, ch, color, origin_x, origin_y, font=font, font_scale=font_scale, font_thickness=font_thickness)
            origin_x += width_ch

        # block = cv2.putText(block, line, origin, FONT, fontScale=FONT_SCALE, color=(255, 255, 255), thickness=FONT_THICKNESS, lineType=cv2.LINE_AA)
        line_blocks.append(block)
        color_idx += len(line)

    final_img = np.concatenate([img] + line_blocks, 0)
    return final_img


def get_video_frames(frames_dir):
    # frames_dir = osp.join(get_video_frames_dir(), f"{video_id:06d}")
    path_list = sorted(glob(osp.join(frames_dir, "*.jpg")))
    assert path_list, f"No frames directory found at {frames_dir}"
    images = []
    for p in path_list:
        images.append(cv2.imread(p, cv2.IMREAD_COLOR))
        assert images[-1] is not None, f"Error reading image at: {p}"

    return images


def viz_worker_fn(args):
    img, segs_t, lines, img_h, img_w, caption_str, char_colors, save_path = args
    cmap = create_color_map().tolist()

    for track_id, mask_rle in zip(segs_t['track_ids'], segs_t['mask_rles']):
        mask_arr = rle_to_mask(mask_rle)
        img = annotate_image_instance(img, mask_arr, color=cmap[track_id], mask_opacity=0.4, mask_border=5)

    # pad on the right if the caption is too long and will take too many lines
    if len(lines) > MAX_LINES_EN:
        width_padding = img_w * ((len(lines) / float(MAX_LINES_EN)) - 1.0)
        width_padding = int(round(width_padding))
        img = cv2.copyMakeBorder(img, 0, 0, 0, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        padding = np.ones((img_h, width_padding, 3), np.uint8) * 255
        img = np.concatenate((img, padding), 1)

    img = cv2.copyMakeBorder(img, 0, 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img = apply_caption_to_img(img, caption_str, char_colors)

    cv2.imwrite(save_path, img)


def generate_video_viz(
        caption_raw: str,
        json_content_segmentations: List[Dict[str, Any]], 
        video_frames_dir: str, 
        image_size: Tuple[int, int],
        labeled_frames_output_dir: str, 
        labeled_video_output_path: str,
        num_workers: int = 8, 
        progress_bar: bool = True, 
        framerate: int = 10, 
        crf: int = 18, 
        ignore_crowd_id: bool = True, 
        t_offset: int = 0
    ):
    caption_obj = parse_caption(caption_raw)
    existing_frames = glob(osp.join(labeled_frames_output_dir, "*.jpg"))
    for fpath in existing_frames:
        os.remove(fpath)
    os.makedirs(labeled_frames_output_dir, exist_ok=True)

    caption_str = caption_obj.parsed
    char_colors = [[0, 0, 0] for _ in range(len(caption_str))]

    cmap = create_color_map().tolist()
    for obj in caption_obj.objects:
        if -1 in obj.ids and ignore_crowd_id:
            continue

        obj_colors = [cmap[iid] for iid in obj.ids]
        # color_i = cmap[i]
        for i, j in enumerate(range(obj.start_idx, obj.end_idx + 1)):
            color_idx = i % len(obj_colors)
            char_colors[j] = obj_colors[color_idx]

    img_h, img_w = image_size
    lines = split_caption_into_lines(caption_str, img_w, font=FONT, font_scale=FONT_SCALE, font_thickness=FONT_THICKNESS)

    video_frames = get_video_frames(video_frames_dir)
    # output_frames = []

    worker_args = []
    for t, img in enumerate(video_frames[t_offset:], t_offset):
        segs_t = json_content_segmentations[t]
        save_path_t = osp.join(labeled_frames_output_dir, f"{t:04d}.jpg")
        worker_args.append((img, segs_t, lines, img_h, img_w, caption_str, char_colors, save_path_t))

    if num_workers <= 0:
        for args_per_iter in tqdm(worker_args):
            viz_worker_fn(args_per_iter)

    else:
        with mp.Pool(num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(viz_worker_fn, worker_args, chunksize=1), total=len(worker_args), disable=not progress_bar):
                pass

    assert labeled_video_output_path.endswith(".mp4")
    frames_to_video(labeled_frames_output_dir, labeled_video_output_path, extension="jpg", framerate=framerate, crf=crf)


def visualize_lgvis_prompts(prompt, masks, filenames, video_frames_base_dir, show_progress_bar=False, font_scale=1.2, font_thickness=2):
    viz_images = []
    assert len(filenames) == len(masks)
    cmap = create_color_map().tolist()

    for fname_t, mask_t in zip(tqdm(filenames, disable=not show_progress_bar), masks):
        image_path = osp.join(video_frames_base_dir, fname_t)
        assert osp.exists(image_path), f"Image file not found: {image_path}"
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        for i, obj_mask in enumerate(mask_t, 1):
            image = annotate_image_instance(image, obj_mask, color=cmap[i], mask_border=5, mask_opacity=0.4)

        image = apply_caption_to_img(image, prompt, colors=[[0, 0, 0] for _ in range(len(prompt))], font_scale=font_scale, font_thickness=font_thickness)
        viz_images.append(image)

    return viz_images
