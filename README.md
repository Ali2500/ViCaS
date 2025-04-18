# ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation (CVPR'25)

[Ali Athar](https://www.aliathar.net/), [Xueqing Deng](https://sites.google.com/view/xueqingdeng7/home), [Liang-Chieh Chen](http://liangchiehchen.com/)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://ali2500.github.io/vicas-project/)
[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)](https://huggingface.co/datasets/Ali2500/ViCaS)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2412.09754)
[![Full Paper](https://img.shields.io/badge/Full_Paper-Read-0000FF.svg)](https://arxiv.org/pdf/2412.09754)

<p align="center" width="120%">
    <img src="https://github.com/Ali2500/ViCaS/blob/main/assets/teaser.gif">
</p>

## 🔥 Highlights

1. We introduce ViCaS, a **human-annotated video dataset** containing **thousands of videos with detailed captions, along with pixel-precise segmentation masks** for salient objects with phrase-grounding to the caption.
2. Our benchmark contains two tasks: **(a) Video Captioning**, which evaluates high-level video understanding, and **(b) Language-Guided Video Instance Segmentation (LGVIS)**, which evaluates finegrained, pixel-level localization based on text prompts.
3. We propose **Video-LLaVA-Seg**, an effective baseline architecture that can tackle both of our benchmark tasks with a single, end-to-end trained model.

##  :loudspeaker: Updates

- **14 Mar 2025:** Uploaded v1.0 of the dataset with annotations for 20,416 videos.
- **26 Feb 2025:** We're accepted to CVPR 2025.
- **12 Dec 2024:** Uploaded v0.1 of the dataset with annotations for 7,331 videos.

## 👨‍💻 Todo

- [X] Release Video-LLaVA-Seg code
- [X] Release Video-LLaVA-Seg model weights

## :hammer: Environment Setup

- Install the required packages:

```
pip3 install -r requirements.txt
```

- Install ffmpeg

```
sudo apt install ffmpeg
```

## :movie_camera: Demo 

You can visualize a few samples without downloading the whole dataset. We provide a few example videos under `demo_data/videos`. First, decode these videos into image frames by running:

```bash
bash demo_data/video_to_frames.sh
```

The frames will be saved to `demo_data/video_frames`. Then you can either run the [Jupyter notebook](https://github.com/Ali2500/ViCaS/blob/main/dataset_demo.ipynb) or the equivalent Python script [dataset_demo.py](https://github.com/Ali2500/ViCaS/blob/main/dataset_demo.py)

## :arrow_double_down: Dataset Download

The annotations are hosted on [HuggingFace](https://huggingface.co/datasets/Ali2500/ViCaS). Clone the HF repo to a directory which we will call `$VICAS_DIR`. Due to copyright reasons, we only provide the annotations (captions and segmentation masks). You have two options to obtain the videos:

**Option 1: Download preprocessed videos**

[Alexey Nekrasov](https://nekrasov.dev/) from the research community has been working with the dataset and was kind enough to upload his preprocessed data to [HuggingFace](https://huggingface.co/datasets/kumuji/ViCaS). In this repo, the videos are split across multiple directories, but you should put them all in a single directory called `videos` and place this directory under `$VICAS_DIR`.

**Option 2: Download and preprocess the videos**

Download the Oops dataset videos from [here](https://omnomnom.vision.rwth-aachen.de/data/PointVOS/videos/Oops/) and put them under some directory `$OOPS_VIDEOS_DIR` with `train` and `val` subdirectories. Then, run the preprocessing script:

```bash
python3 vicas/preprocess/gather_videos.py --vicas_dir $VICAS_DIR --oops_dir $OOPS_VIDEOS_DIR
```

This will create a directory at `$VICAS_DIR/videos` and put the required videos there with the video IDs prepended to the filename. 

### Decode Video Frames

To train and evaluate LG-VIS, you also need to decode the videos into image frames:

```bash
bash vicas/preprocess/videos_to_frames.sh $VICAS_DIR/videos $VICAS_DIR/video_frames
```

The image frames for each video will be saved to a directory at `$VICAS_DIR/video_frames/<video_id>`.

### File Structure

Once the videos are downloaded and decoded into frames, the file structure for `$VICAS_DIR` should look like this:

```
$VICAS_DIR
├── videos                      
│   ├── <video #1.mp4>
│   ├── <video #2.mp4>
│   ├── ...
├── video_frames
│   ├── <video #1>
│   │   └── 00000.jpg
│   │   └── 00001.jpg
│   │   └── ...
│   ├── <video #2>
│   │   └── 00000.jpg
│   │   └── 00001.jpg
│   ├── ...
├── annotations               
│   ├── v0.1
│   │   └── <video #1.json>
│   │   └── <video #2.json>
│   │   └── ...
├── splits
│   ├── v0.1
│   │   └── train.json
│   │   └── val.json
│   │   └── test.json
```

## Annotation Format

We provide an easy-to-use API under `vicas/dataset.py` to parse the dataset and its JSON annotations. Please look at the `ViCaSVideo` class definition to see the JSON fields should be parsed. Refer to the [Jupyter notebook](dataset_demo.ipynb) or [Python demo](dataset_demo.py) to see various use-cases for the API.

### TL;DR for Captions Only

If you're only interested in the captions, just use the `caption_parsed_en_gpt` value in the annotation file:

```python
import json
with open("<VICAS_DIR>/annotations/v0.1/00000.json") as fh:
    content = json.load(fh)
caption = content["caption_parsed_en_gpt"]
```

### Chinese Captions

We also provide captions in Chinese. These were obtained by machine-translating the English captions. Note that no quality checks were applied to them so they may contain errors.

```python
from vicas.caption_parsing import parse_caption

caption_raw = content["caption_raw_cn"]             # with object-grounding syntax
caption_parsed = parse_caption(caption_raw).parsed  # regular, human-readable caption
```

## Benchmark Evaluation

The predictions are in a per-video JSON format similar to the ground-truth. A set of ~1000 prediction files is provided in the HF repo for reference. In short, each JSON file needs to have the following fields `video_id`, `pred_caption` and `pred_lgvis_masks`. You can inspect the example predictions to see the exact format.

Evaluate captioning accuracy requires Llama3-70B. Refer to the [offical website](https://www.llama.com/llama-downloads/) to download the `Meta-Llama-3-70B-Instruct` model checkpoint. We use the original (3.0) model version. You will need 8 GPUs to run this model. We will call the checkpoint directory `$LLAMA3_MODEL_DIR` and it should contain `tokenizer.model` and several `.pth` files. With this setup, you can the run the evaluation script as follows:

```bash
bash vicas/evaluation/run.sh --pred_dir /path/to/pred --gt_dir /path/to/gt --llama_ckpt_dir $LLAMA3_MODEL_DIR --split {val,test} -o /path/to/eval_output.json
```

The calculated scores will be printed and also written to `/path/to/eval_output.json`

### Task-specific Evaluation

The following instructions are for evaluating just one of the tasks. In this case, the prediction files need not contain . Note that LG-VIS evaluation does not require any GPUs.

- **Video Captioning Only:**

```bash
torchrun --nproc_per_node=8 --master_port 2222 vicas/evaluation/main.py --pred_dir /path/to/pred --gt_dir $VICAS_DIR/annotations/v0.1 --llama_ckpt_dir $LLAMA3_MODEL_DIR --split {val,test} --skip_masks -o /path/to/eval_output.json
```

- **LG-VIS Only**:

```bash
python3 vicas/evaluation/main.py --pred_dir /path/to/pred --gt_dir $VICAS_DIR/annotations/v0.1 --split {val,test} --skip_captions -o /path/to/eval_output.json
```

For further details about the launch arguments for the eval script, run `python3 vicas/evaluation/main.py --help`.

## Video-LLaVA-Seg Baseline

We have open-sourced the code for our baseline as well as trained model weights in a [separate repository](https://github.com/Ali2500/Video-LLaVA-Seg). Please refer to the instructions there to setup and run it.

## ⚠️ Terms of use
* This dataset cannot be used for commercial purposes. It has been created for research purposes only.
* This is not an official ByteDance product.

## BibTeX

```
@article{athar2024vicas,
author = {Ali Athar, Xueqing Deng, Liang-Chieh Chen},
title = {ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation},
journal = {CVPR},
year = {2025}
}
```

## :sparkles: Acknowledgements

- [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT): We built the Video-LLaVA-Seg codebase based on this.
- [SAM2](https://github.com/facebookresearch/sam2/tree/main): We adopted this model as our segmentation network 
