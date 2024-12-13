# ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation

[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)](https://huggingface.co/datasets/Ali2500/ViCaS)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://ali2500.github.io/vicas-project/)

<p align="center" width="100%">
    <img src="https://github.com/Ali2500/ViCaS/blob/main/assets/teaser.gif">
</p>

## Updates

- 12 Dec 2024: Uploaded v0.1 of the dataset with 7,331 videos.

## Demo

You can visualize a few samples without downloading the whole dataset. We provide a few example videos under `demo_data/videos`. First, decode these videos into image frames by running:

```bash
bash demo_data/video_to_frames.sh
```

Then you can either run the [Jupyter notebook](https://github.com/Ali2500/ViCaS/blob/main/dataset_demo.ipynb) or the equivalent Python script [dataset_demo.py](https://github.com/Ali2500/ViCaS/blob/main/dataset_demo.py)

## Dataset Download

The annotations are hosted on [HuggingFace](https://huggingface.co/datasets/Ali2500/ViCaS). Clone the HF repo to a directory which we will call `$VICAS_DIR`.

Due to copyright reasons, we only provide the annotations (captions and segmentation masks). Download the Oops dataset videos from [here](https://omnomnom.vision.rwth-aachen.de/data/PointVOS/videos/Oops/) and put them under a directory `$OOPS_VIDEOS_DIR` with `train` and `val` subdirectories. Then, run the preprocessing script:

```bash
python3 vicas/preprocess/gather_videos.py --vicas_dir $VICAS_DIR --oops_dir $OOPS_VIDEOS_DIR
```

This will create a directory at `$VICAS_DIR/videos` and put the required videos there with the video IDs prepended to the filename. **You need ffmpeg installed** for this step since some videos need to be split because of cutscenes.

Once this is done, you're all set. The file structure should look like this:

```
$VICAS_DIR
├── videos                      
│   ├── <video #1.mp4>
│   ├── <video #2.mp4>
│   ├── <video #... >
├── annotations               
│   ├── v0.1
│   │   └── <video #1.json>
│   │   └── <video #2.json>
│   │   └── <video #... >
```

## Annotation Format

We provide an easy-to-use API under `vicas/dataset.py` to parse the dataset and its JSON annotations. Please look at the `ViCaSVideo` class definition to see the JSON fields should be parsed. Refer to the Jupyter notebook or Python demo to see various use-cases for the API.

#### TL;DR for Captions Only

If you're only interested in the captions, just use the `caption_parsed_en_gpt` value in the annotation file:

```python
import json
with open("<VICAS_DIR>/annotations/v0.1/00000.json") as fh:
    content = json.load(fh)
caption = content["caption_parsed_en_gpt"]
```

## Benchmark Evaluation

The predictions are in a per-video JSON format similar to the ground-truth. A set of ~1000 prediction files is provided in the HF repo for reference. In short, each JSON file needs to have the following fields `video_id`, `pred_caption` and `pred_lgvis_masks`. You can inspect the example predictions to see the exact format.

Evaluate captioning accuracy requires Llama3-70B. Refer to the [offical website](https://www.llama.com/llama-downloads/) to download the model checkpoint. We use the original (3.0) model version. You will need 8 GPUs to run this model. We will call the checkpoint directory `$LLAMA3_MODEL_DIR` and it should contain `tokenizer.model` and several `.pth` files. You can the run the evaluation script as follows:

```bash
bash vicas/evaluation/run.sh --pred_dir /path/to/pred --gt_dir /path/to/gt --llama_ckpt_dir $LLAMA3_MODEL_DIR --split {val,test}
```

#### Task-specific Evaluation

If you're only interested in one of the tasks, you can completely omit the annotations for the other task from the prediction files and run the evaluation as follows. Note that LG-VIS evaluation does not require any GPUs.

- Video Captioning Only:

```bash
torchrun --nproc_per_node=8 --master_port 2222 vicas/evaluation/main.py --pred_dir /path/to/pred --gt_dir $VICAS_DIR/annotations/v0.1 --llama_ckpt_dir $LLAMA3_MODEL_DIR --split {val,test} --skip_masks
```

- LG-VIS Only:

```bash
python3 vicas/evaluation/main.py --pred_dir /path/to/pred --gt_dir $VICAS_DIR/annotations/v0.1 --split {val,test} --skip_captions
```

For further details about the launch arguments for the eval script, run `python3 vicas/evaluation/main.py --help`.

## BibTeX

```
@article{athar2024vicas,
author = {Ali Athar, Xueqing Deng, Liang-Chieh Chen},
title = {ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation},
journal = {Arxiv},
year = {2024}
}
```
