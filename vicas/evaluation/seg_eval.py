from typing import List, Dict, Optional, Any, Union

import numpy as np
import pycocotools.mask as mt

from vicas.evaluation.utils import compute_track_ious


class TrackMAP:
    def __init__(self, threshs: Optional[List[int]] = None):
        if threshs is None:
            self.threshs = np.arange(0.5, 0.95 + 1e-4, 0.05)
        else:
            self.threshs = np.array(threshs)

        self.max_tracks_per_video = 10
        self.threshs_to_report = [0.5, 0.75, 0.9]

        self.thresh_idxes_to_report = []
        for i, thresh in enumerate(self.threshs_to_report):
            match = np.abs(self.threshs - thresh) < 1e-4
            assert np.sum(match) == 1, f"Threshold {thresh} has to be reported, but it was not found in the list of thresholds"
            self.thresh_idxes_to_report.append(match.tolist().index(1))
        self.thresh_idxes_to_report = np.array(self.thresh_idxes_to_report)

    def process_video(
            self, 
            pred_tracks: List[List[Dict[str, Any]]], 
            gt_tracks: List[List[Dict[str, Any]]],
            pred_scores: Optional[Union[np.ndarray, List[float]]] = None
        ) -> Dict[str, Any]:

        if isinstance(pred_scores, (list, tuple)):
            pred_scores = np.array(pred_scores, np.float32)
        elif pred_scores is None:
            pred_scores = np.ones(len(pred_tracks), np.float32)
        else:
            assert isinstance(pred_scores, np.ndarray)

        num_pred = len(pred_tracks)
        assert list(pred_scores.shape) == [num_pred]
        num_gt = len(gt_tracks)
        num_thresh = len(self.threshs)

        if num_pred == 0:
            match_gt_idx = np.zeros((num_thresh, 0), np.int32)
        elif num_gt == 0:
            match_gt_idx = np.full((num_thresh, num_pred), -1, np.int32)
        else:
            # compute assignment between pred and GT
            sort_idx = np.argsort(-1.0 * pred_scores)[:self.max_tracks_per_video]
            pred_tracks = [pred_tracks[i] for i in sort_idx.tolist()]
            pred_scores = pred_scores[sort_idx]

            pred_gt_ious = compute_track_ious(pred_tracks, gt_tracks)
            match_gt_idx = np.full((num_thresh, num_pred), -1, np.int32)

            for lvl, thresh in enumerate(self.threshs):
                for i in range(num_pred):
                    for j in range(num_gt):
                        if j in match_gt_idx[lvl]:
                            continue
                        
                        if pred_gt_ious[i, j] >= thresh:
                            match_gt_idx[lvl, i] = j

        return {
            "match_indices": match_gt_idx,
            "scores": pred_scores,
            "num_gt": num_gt
        }
    
    def combine_video_results(self, video_level_results: List[Dict[str, Any]]):
        match_indices = np.concatenate([x['match_indices'] for x in video_level_results], 1)  # [num_thresh, total_pred_tracks]
        scores = np.concatenate([x['scores'] for x in video_level_results]) # [total_pred_tracks]
        num_gt = float(sum([x['num_gt'] for x in video_level_results]))

        if match_indices.size == 0:
            print(f"WARN: There are zero predicted tracks")
            aps = np.zeros((len(self.threshs),), np.float32)
        elif num_gt == 0:
            print(f"WARN: There are zero ground-truth tracks.")
            aps = np.ones((len(self.threshs),), np.float32)
        else:
            sort_idx = np.argsort(-1.0 * scores)  # [total_pred_tracks]
            match_indices = np.stack([
                match_indices_per_thresh[sort_idx] 
                for match_indices_per_thresh in match_indices
            ])  # [num_thresh, total_pred_tracks]

            tps = match_indices != -1  # [num_thresh, total_pred_tracks]
            fps = match_indices == -1  # [num_thresh, total_pred_tracks]

            tp_sum = np.cumsum(tps.astype(np.float32), 1) 
            fp_sum = np.cumsum(fps.astype(np.float32), 1)

            precision = tp_sum / (tp_sum + fp_sum + 1e-8)
            recall = tp_sum / num_gt

            aps = self._pr_to_ap(precision, recall)
        
        aps = aps * 100.0  # scale to [0, 100]
        aps_to_report = aps[self.thresh_idxes_to_report]

        ret_dict = {
            "mAP": np.mean(aps).item()
        }

        for thresh, thresh_ap in zip(self.threshs_to_report, aps_to_report):
            ret_dict[f"AP@{int(thresh*100)}"] = thresh_ap.item()

        return ret_dict

    def _pr_to_ap(self, precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
        # precision, recall: [num_threshs, num_tracks]
        assert precision.shape == recall.shape, f"Shape mismatch: {precision.shape}, {recall.shape}"
        zero_pad = np.zeros_like(precision[:, :1])
        precision = np.concatenate([zero_pad, precision, zero_pad], 1)
        recall = np.concatenate([zero_pad, recall, np.ones_like(zero_pad)], 1)

        aps = np.zeros(len(self.threshs), np.float32)
        for thresh_idx, (prec_per_thresh, rec_per_thresh) in enumerate(zip(precision, recall)):
            for i in range(len(prec_per_thresh) - 1, 0, -1):
                prec_per_thresh[i - 1] = np.maximum(prec_per_thresh[i - 1], prec_per_thresh[i])

            indices = np.where(rec_per_thresh[1:] != rec_per_thresh[:-1])[0]
            aps[thresh_idx] = np.sum((rec_per_thresh[indices + 1] - rec_per_thresh[indices]) * prec_per_thresh[indices + 1])

        return aps

    # def _per_video_results_to_ar(self, video_level_results: List[Dict[str, Any]]):

