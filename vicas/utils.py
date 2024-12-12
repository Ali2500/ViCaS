from glob import glob

import os
import os.path as osp
import subprocess
import tempfile
import shutil
import random
import string

try:
    import cv2
    CV2_IMPORTED = True
except ImportError as _:
    CV2_IMPORTED = False


def frames_to_video(frames_dir, output_path, framerate, extension="jpg", **kwargs):
    img_paths = sorted(glob(osp.join(frames_dir, f"*.{extension}")))
    if len(img_paths) == 0:
        raise ValueError(f"No files with extension {extension} found in {frames_dir}")
    
    # check if image dims are even
    image = cv2.imread(img_paths[0], cv2.IMREAD_UNCHANGED)
    pad_h = image.shape[0] % 2
    pad_w = image.shape[1] % 2

    if pad_w or pad_h:
        # ffmpeg throws an error if the width or height are not multiples of 2. For such cases we need to
        # pad the images and save them to a temporary directory
        rand_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        src_dir = osp.join(tempfile.gettempdir(), rand_string)
        os.makedirs(src_dir)

        for p in img_paths:
            image = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)
            fname = osp.split(p)[-1]
            cv2.imwrite(osp.join(src_dir, fname), image)

        del_src_dir = True
    else:
        src_dir = frames_dir
        del_src_dir = False

    crf = kwargs.get('crf', 18)
    codec = kwargs.get('codec', 'libx264')

    cmd = [
        'ffmpeg', '-y', '-framerate', str(framerate), '-pattern_type', 'glob', '-i', f"{src_dir}/*.{extension}", 
        '-c:v', codec, '-crf', str(crf), '-pix_fmt', 'yuv420p', output_path
    ]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        cmd_string = " ".join(cmd)
        raise RuntimeError(f"The following command returned {ret.returncode}:\n{cmd_string}\nSTDERR:\n{ret.stderr}")

    if not osp.exists(output_path):
        cmd_string = " ".join(cmd)
        raise RuntimeError(f"The following command ran successfully, but no video file was generated at {output_path}:\n{cmd_string}")

    if del_src_dir:
        shutil.rmtree(src_dir)


def video_to_frames(video_path: str, output_dir: str, fps: int = 30, suppress_output: bool = False):
    if not (osp.exists(output_dir) or osp.islink(output_dir)):
        os.makedirs(output_dir, exist_ok=True)

    cmd = ['ffmpeg', '-y', '-i', video_path, '-r', str(fps), '-q:v', '1', '-start_number', '0', f"{output_dir}/" + r"%05d.jpg"]
    if suppress_output:
        ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    else:
        ret = subprocess.run(cmd).returncode

    if ret != 0:
        raise RuntimeError(f"Failed to run ffmpeg command for {video_path}")
