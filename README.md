# ViCaS: A Dataset for Combining Holistic and Pixel-level Video Understanding using Captions with Grounded Segmentation

[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)](https://huggingface.co/datasets/Ali2500/ViCaS)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)]([https://xdeng7.github.io/coconut.github.io/](https://ali2500.github.io/vicas-project/))

<p align="center" width="100%">
    <img src="https://github.com/Ali2500/ViCaS/blob/main/assets/teaser.gif">
</p>

## Updates

- 12.12.2024: Uploaded v0.1 of the dataset with 7,336 videos.

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
