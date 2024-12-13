from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import json
import torch.distributed as dist
import pycocotools.mask as mt
import numpy as np
import os.path as osp

from vicas.evaluation.caption_eval import CaptionEvalLlama3
from vicas.evaluation.seg_eval import TrackMAP


def is_rank0():
    if dist.is_initialized():
        return dist.get_rank() == 0
    else:
        return True


def print_rank0(args, **kwargs):
    if is_rank0():
        print(args, **kwargs)


def get_null_mask(height, width):
    arr = np.zeros((height, width), np.uint8)
    return mt.encode(np.asfortranarray(arr))


def get_gt_rles(gt_content: Dict[str, Any], frame_filenames: List[str], track_ids: List[int]):
    gt_rles = [[] for _ in range(len(track_ids))]  # outer list over objects, inner list over frames
    segmentations = {segs_t['filename']: segs_t for segs_t in gt_content["segmentations"]}
    height, width = gt_content["image_size"]

    for fname in frame_filenames:
        segs_t = segmentations[fname]
        for i, t_id in enumerate(track_ids):
            if t_id in segs_t["track_ids"]:
                index = segs_t["track_ids"].index(t_id)
                rle = segs_t["mask_rles"][index]
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("utf-8")
                gt_rles[i].append(rle)
            else:
                gt_rles[i].append(get_null_mask(height, width))

    assert all([len(track_rles) == len(frame_filenames) for track_rles in gt_rles])
    return gt_rles


def parse_pred_and_gt_jsons(pred_path, gt_path, skip_masks, skip_captions):
    with open(pred_path, 'r') as fh:
        pred_content = json.load(fh)

    with open(gt_path, 'r') as fh:
        gt_content = json.load(fh)

    if skip_captions:
        pred_caption = gt_captions = None
    else:
        pred_caption = pred_content["pred_caption"]
        gt_captions = [gt_content["caption_parsed_en"], gt_content["caption_parsed_en_gpt"]] + gt_content["reworded_en_captions"]
    
    # parse masks
    if skip_masks:
        pred_mask_list = gt_mask_list = None
    else:
        num_referrals = len(gt_content["object_referrals"])
        pred_mask_list = [[] for _ in range(num_referrals)]
        gt_mask_list = [[] for _ in range(num_referrals)]
        gt_filenames = sorted([segs_t['filename'] for segs_t in gt_content["segmentations"] if segs_t['is_gt']])

        for i in range(num_referrals):
            pred_referral = pred_content["pred_lgvis_masks"][i]
            gt_referral = gt_content["object_referrals"][i]
            pred_rles_i_dict = {}

            for segs_t in pred_referral:
                if segs_t['filename'] not in gt_filenames:
                    continue

                for rle in segs_t["mask_rles"]:
                    rle["counts"] = rle["counts"].encode("utf-8")

                pred_rles_i_dict[segs_t['filename']] = segs_t["mask_rles"]  # List of RLE dicts (one per predicted object)

            if set(pred_rles_i_dict.keys()) != set(gt_filenames):
                raise ValueError(f"Video {gt_content['video_id']}: all ground-truth frames not found in predictions:\nPred: {sorted(list(pred_rles_i_dict.keys()))}\nGT: {gt_filenames}")

            assert len(set([len(x) for x in pred_rles_i_dict.values()])) == 1 
            pred_rles_i = [pred_rles_i_dict[fname] for fname in gt_filenames]
            # at this point, pred_rles_i is a nested list with outer list over time and inner list over tracks. We need to flip this around
            pred_rles_i = list(zip(*pred_rles_i))

            gt_rles_i = get_gt_rles(gt_content, gt_filenames, gt_referral["track_ids"])  # TODO: change the format. Use the global 'track_ids' list for all segs_t

            # sanity checks
            # assert len(pred_rles_i) == len(gt_rles_i)
            assert all([len(x) == len(y) for x, y in zip(pred_rles_i, gt_rles_i)])

            pred_mask_list.append(pred_rles_i)
            gt_mask_list.append(gt_rles_i)

    return {
        "pred_caption": pred_caption,
        "gt_captions": gt_captions,
        "pred_masks": pred_mask_list,
        "gt_masks": gt_mask_list
    }


def main(args):
    if not args.output_path:
        args.output_path = osp.join(args.pred_dir.rstrip("/") + "_eval_output.json")

    print(f"Output will be saved to {args.output_path}")

    split_info_file = Path(__file__).parent.parent.parent.joinpath(f"splits/{args.dataset_version}/{args.split}.json")
    with open(split_info_file, 'r') as fh:
        eval_video_ids = sorted(json.load(fh))

    if args.skip_masks:
        mask_evaluator = per_referral_outputs = None
    else:
        mask_evaluator = TrackMAP()
        per_referral_outputs = []

    if args.skip_captions:
        pred_captions = gt_captions = None
    else:
        pred_captions = []
        gt_captions = []

    for video_id in tqdm(eval_video_ids):
        pred_path = osp.join(args.pred_dir, f"{video_id:06d}.json")
        assert osp.exists(pred_path), f"No prediction file found at: {pred_path}"

        gt_path = osp.join(args.gt_dir, f"{video_id:06d}.json")
        if not osp.exists(gt_path):
            # TODO: deprecate this mechanism. GT json files should be saved as <video_id>.json
            gt_path = glob(gt_path.replace(".json", "_*.json"))
            assert len(gt_path) == 1, f"Should only be 1 match, but got {gt_path}"
            gt_path = gt_path[0]

        assert osp.exists(pred_path), f"No GT file found at: {gt_path}"

        # load and parse pred and GT
        data = parse_pred_and_gt_jsons(pred_path, gt_path, skip_masks=args.skip_masks, skip_captions=args.skip_captions)

        if mask_evaluator is not None:
            for pred_masks_per_ref, gt_masks_per_ref in zip(data["pred_masks"], data["gt_masks"]):
                per_referral_outputs.append(mask_evaluator.process_video(pred_masks_per_ref, gt_masks_per_ref))

        if pred_captions is not None:
            pred_captions.append(data["pred_caption"])
            gt_captions.append(data["gt_captions"])

    results_dict = {}

    if not args.skip_masks:
        mask_results = mask_evaluator.combine_video_results(per_referral_outputs)
        results_dict["LG-VIS"] = mask_results
        print_rank0(f"LG-VIS Metrics: {mask_results}")

    if not args.skip_captions:
        caption_evaluator = CaptionEvalLlama3(checkpoint_dir=args.llama_ckpt_dir, batch_size=1, pooling="avg")
        scores = caption_evaluator.run(pred_captions, gt_captions)
        mean = sum(scores) / float(len(scores))
        results_dict[f"Captioning"] = mean
        print_rank0(f"Caption accuracy: {mean}")

    mode = 'a' if args.append_metrics else 'w'

    if is_rank0():
        with open(args.output_path, mode) as fh:
            fh.write(json.dumps(results_dict, indent=2) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="ViCaS Benchmark Evaluation Script")

    parser.add_argument("--pred_dir", "-i", required=True, help="Path to directory containing per-video JSON files with predicted captions and segmentation masks")
    parser.add_argument("--gt_dir", required=True, help="Path to directory containing ground-truth JSON annotations")
    parser.add_argument("--output_path", "-o", required=True, help="Path to JSON file where the evaluation output will be saved.")

    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Specify which split to evaluate")
    parser.add_argument("--dataset_version", default="0.1", help="Specify which version of the dataset to evaluate")

    parser.add_argument("--llama_ckpt_dir", required=False, help="Path to Llama3-70B checkpoint directory. This directory should contain a tokenizer.model file and several *.pth files. Not needed if '--skip_captions' is set.")
    parser.add_argument("--skip_captions", action='store_true', help="If set, only the LG-VIS metrics will be calculated and the predicted captions will be ignored")
    parser.add_argument("--skip_masks", action='store_true', help="If set, only captioning accuracy will be calculated and segmentation predictions will be ignored.")

    parser.add_argument("--append_metrics", action='store_true', help="If set, the calculated metrics will be appended to '--output_path' if it exists.")

    main(parser.parse_args())
