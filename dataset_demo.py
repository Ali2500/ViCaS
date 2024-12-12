import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile

from vicas.dataset import ViCaSDataset, ViCaSVideo
from vicas.caption_parsing import parse_caption


# Set this variable:
annotations_dir = "/path/to/vicas/json/annotations/dir"

video_frames_dir = "demo_data/video_frames"
split = None # can be set to 'train', 'val' or 'test' to load a particular split

dataset = ViCaSDataset(
    annotations_dir, 
    split=split,
    video_frames_dir=video_frames_dir
)
print(f"Indexed {len(dataset)} videos from the dataset")

# Parse a video
example_video_id = 9505  # videos for a few example videos are provided in 'demo_data'.
video = dataset.parse_video(example_video_id) # can also run: ViCaSVideo.from_json(f"{annotations_dir}/{example_video_id:06d}.json")

# Create a temporary directory to store the visualization
viz_temp_dir = os.path.join(tempfile.gettempdir(), "ViCaS_demo", f"{example_video_id:06d}")
os.makedirs(viz_temp_dir, exist_ok=True)
print(f"Visualization output will be saved to: {viz_temp_dir}")

# After running this, visualization video will be saved to <viz_temp_dir>/video.mp4
video.visualize(viz_temp_dir)

# Print captions
print("Caption with phrase-grounding syntax: " + video.caption_gpt_raw)
print("Parsed caption without grounding syntax: " + video.caption_gpt_parsed)

# Parse the caption and return a VideoCaption object
caption_obj = parse_caption(video.caption_gpt_raw)
print("Parsed caption without grounding syntax: " + caption_obj.parsed)  # same as the parsed caption printed above

# Pretty print the caption with grounding-phrases
print(caption_obj)

# Parse LG-VIS prompts
print(f"This video contains {video.num_lgvis_prompts} LG-VIS prompts")

# Iterate over LG-VIS prompts and display a few visualization frames 
for prompt, masks, track_ids, filenames, viz_frames in video.parse_lgvis(return_viz=True): # iterate over prompts
    print("Prompt: " + prompt)
    print(f"There are {len(masks)} frame-level masks")
    print(f"This prompt references {len(masks[0])} object tracks with IDs {track_ids}")
    print(f"Each mask array has shape {masks[0][0].shape} and dtype {masks[0][0].dtype}")
    
    # display 6 frames
    frame_indices = np.linspace(0, len(viz_frames)-1, 6).astype(int).tolist()
    viz_frames = [viz_frames[i] for i in frame_indices]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(viz_frames[i][:, :, ::-1]) # convert image BGR to RGB
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
