from typing import List, Dict, Optional, Any, Union

import copy
import numpy as np
import pycocotools.mask as mt
import torch.distributed as dist


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0


def compute_track_ious(x: List[List[Dict[str, Any]]], y: List[List[Dict[str, Any]]]):
    # x,y: outer list: over tracks, inner list: over frames. Each entry is an RLE dict
    # returns a matrix of size [len(x), len(y)] with track IoUs between x and y
    num_x = len(x)
    num_y = len(y)

    ious = np.zeros((num_x, num_y), np.float32)
    for i in range(num_x):
            track_i = x[i]
            num_frames = len(track_i)

            for j in range(num_y):
                track_j = y[j]
                assert len(track_j) == num_frames, f"Predicted track has length {num_frames} which is different from the length of the ground-truth track {len(track_j)}" 

                intersection = 0.
                union = 0.
                for t in range(num_frames):
                    rle_i_t = track_i[t]
                    rle_j_t = track_j[t]

                    if isinstance(rle_i_t, str):
                        rle_i_t = copy.deepcopy(rle_i_t)
                        rle_i_t["counts"] = rle_i_t["counts"].encode("utf-8")
                    if isinstance(rle_j_t, str):
                        rle_j_t = copy.deepcopy(rle_j_t)
                        rle_j_t["counts"] = rle_j_t["counts"].encode("utf-8")

                    assert "size" in rle_i_t and "counts" in rle_i_t
                    assert "size" in rle_j_t and "counts" in rle_j_t

                    if rle_i_t and rle_j_t:
                        intersection += mt.area(mt.merge([rle_i_t, rle_j_t], True))
                        union += mt.area(mt.merge([rle_i_t, rle_j_t], False))
                    elif rle_i_t and not rle_j_t:
                        union += mt.area(rle_i_t)
                    elif not rle_i_t and rle_j_t:
                        union += mt.area(rle_j_t)

                if union < 0.0 - np.finfo('float').eps:
                    raise RuntimeError(f"Union value = {union} which is <0.")
                if intersection > union:
                    raise RuntimeError(f"Intersection value ({intersection}) > union value ({union})")
                
                ious[i, j] = intersection / union if union > 0.0 + np.finfo('float').eps else 0.0 

    return ious


def compute_track_ious_numpy(x: np.ndarray, y: np.ndarray):
    # x, y: [N/M, T, H, W]
    assert x.shape[1:] == y.shape[1:]

    if x.shape[0] == 0:
        return np.zeros((0, y.shape[0]), np.float32)
    elif y.shape[0] == 0:
        return np.zeros((x.shape[0], 0), np.float32)

    x = np.reshape(x, (x.shape[0], -1))  # [N, pts]
    y = np.reshape(y, (y.shape[0], -1))  # [M, pts]

    x = x[:, None, :]  # [N, 1, pts]
    y = y[None, :, :]  # [1, M, pts]

    inter = np.logical_and(x, y).astype(np.float32).sum(2)
    union = np.logical_or(x, y).astype(np.float32).sum(2)

    return inter / (union + 1e-8)
