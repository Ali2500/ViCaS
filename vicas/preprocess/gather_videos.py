import os
import os.path as osp
import json
import subprocess
import shutil

from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm


OOPS_VIDEOS_DIR = "" # should have 'train' and 'val' subdirs
VICAS_VIDEOS_DIR = ""  # video files will be copied from the source dataset to this path
VICAS_JSON_ANNOTATIONS_DIR = ""  # directory with JSON annotations. Needed for reference; nothing will be written here
UVO_SPARSE_VIDEOS_DIR = ""  # dir containing UVO-preprocessed sparse videos.
KINETICS_VIDEOS_DIR = ""  # should contain 700 category directories


def process_oops_scene_video(video_dict):
    # some Oops videos need to be split to remove cutscenes
    start_time = video_dict['scene_spec']['start']
    end_time = video_dict['scene_spec']['end']
    src_path = osp.join(OOPS_VIDEOS_DIR, video_dict["split"], video_dict["orig_dataset_filename"])
    assert osp.exists(src_path), f"Video not found at {src_path}"

    tgt_path = osp.join(VICAS_VIDEOS_DIR, video_dict["vicas_filename"])
    cmd = ['ffmpeg', '-y', '-i', src_path, "-ss", start_time, "-t", end_time, "-c:v", "libx264", "-crf", "15", "-c:a", "aac", tgt_path]

    ret = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if ret != 0:
        cmd_string = " ".join(cmd)
        raise RuntimeError(f"Failed to split video at {src_path}. Try running the following command in terminal to debug the issue:\n{cmd_string}")


def main(args):
    global OOPS_VIDEOS_DIR, UVO_SPARSE_VIDEOS_DIR, KINETICS_VIDEOS_DIR, VICAS_VIDEOS_DIR, VICAS_JSON_ANNOTATIONS_DIR

    OOPS_VIDEOS_DIR = args.oops_dir
    VICAS_VIDEOS_DIR = osp.join(args.vicas_dir, "videos")
    VICAS_JSON_ANNOTATIONS_DIR = osp.join(args.vicas_dir, "annotations", args.dataset_version)
    UVO_SPARSE_VIDEOS_DIR = args.uvo_dir
    KINETICS_VIDEOS_DIR = args.kinetics_dir

    preprocess_specs_file = osp.join(osp.dirname(__file__), "video_preprocess_specs.json")
    assert osp.exists(preprocess_specs_file)

    with open(preprocess_specs_file, 'r') as fh:
        content = json.load(fh)

    content = {d['video_id']: d for d in content}

    json_files = glob(osp.join(VICAS_JSON_ANNOTATIONS_DIR, "*.json"))
    video_ids_to_process = sorted([int(osp.split(p)[-1].replace(".json", "")) for p in json_files])

    os.makedirs(VICAS_VIDEOS_DIR, exist_ok=True)

    for video_id in tqdm(video_ids_to_process):
        video_dict = content[video_id]

        if video_dict["src_dataset"] == "oops":
            if "scene_spec" in video_dict:
                process_oops_scene_video(video_dict)
                continue
            else:
                src_path = osp.join(OOPS_VIDEOS_DIR, video_dict["split"], video_dict["orig_dataset_filename"])

        elif video_dict["src_dataset"] == "uvo":
            src_path = osp.join(UVO_SPARSE_VIDEOS_DIR, video_dict["orig_dataset_filename"])

        elif video_dict["src_dataset"] == "kinetics":
            src_path = osp.join(KINETICS_VIDEOS_DIR, video_dict["kinetics_category"], video_dict["orig_dataset_filename"])

        else:
            raise ValueError(f"Unexpected source dataset: {video_dict['src_dataset']}")
        
        assert osp.exists(src_path), f"Video not found at {src_path}"
        tgt_path = osp.join(VICAS_VIDEOS_DIR, video_dict["vicas_filename"])
        shutil.copy(src_path, tgt_path)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--oops_dir", required=True)
    parser.add_argument("--vicas_dir", required=True)
    parser.add_argument("--uvo_dir", required=False)
    parser.add_argument("--kinetics_dir", required=False)

    parser.add_argument("--dataset_version", default="v0.1")

    main(parser.parse_args())
