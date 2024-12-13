import torch
import math

from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any, Union, Iterable
from vicas.evaluation.llama3 import Llama3Interface
from vicas.evaluation.utils import is_main_process


SYSTEM_MSG = """
You are a language assistant helping the user to assign an accuracy score between two text passages. Generate only the required output without any preamble or header.
"""

USER_MSG = """
You are given two captions for a certain video sequence. The first is the 'ground-truth' which accurately describes what is happening in the video. The second is a machine-generated 'predicted' caption. 
Assign a score of between 0 and 5 to the predicted caption. A score of 0 means the prediction is completely unrelated to the ground-truth and shares no common information. A score of 5 means that the content is conveyed with exactly the same information. Pay attention to the following guidelines when scoring:

1) Focus on the underlying meaning rather than word-to-word similarity. If the overall meaning is similar, then a high score should be assigned even if the words used are different. Conversely, if the meaning differs then a low score should be assigned even if similar words are used.

2) Do not penalize the prediction if it adds or omits minor details that are not relevant to the main event(s) of the video. 
    - Example: If the prediction mentions trees in the background but the ground-truth contains no information about the surroundings then don't penalize the prediction. 
    - If the ground-truth mentions fine details that are not essential to the main content then don't penalize the prediction for omitting this information.

3) Do penalize any missing information that is important to the main events of the video. Also penalize any major events that are in the prediction but not in the ground-truth since these are probably false.

4) Do penalize any information that contradicts the ground-truth
    - Example: If the prediction mentions a person boarding a black truck but the ground-truth states mentions a person entering a white car then the prediction should be penalized for this incorrect information.

5) Do penalize any information about the audio or sounds in the scene. Such information is hallucinated since the machine is not given any audio input and should be moderately penalized. 

The two captions are given below. Output ONLY the accuracy score and nothing else.

GROUND-TRUTH: {}\n\n
PREDICTED: {}
"""


class CaptionEvalLlama3:
    def __init__(self, checkpoint_dir: str, batch_size: int = 8, pooling: str = "max"):
        self.batch_size = batch_size
        assert torch.cuda.device_count() >= 8, f"Llama3-70B needs 8 GPUs for inference"

        self.model = Llama3Interface(ckpt_dir=checkpoint_dir, max_batch_size=batch_size)
        self.name = "Llama3"
        assert pooling in ("max", "avg")
        self.pooling = pooling

    def run(self, preds: List[str], gts: List[Union[str, List[str]]])  -> List[float]:
        """
        preds: List of predicted captions
        gts: List of ground-truth captions. Each list element can, in turn, be a list of multiple correct ground-truths. 
             In this case, the maximum across all ground-truths will be returned

        Returns:
            Scores as a list of length N, where N = len(preds) = len(gts)
        """
        assert len(preds) == len(gts)

        preds_flat = []
        gts_flat = []
        scores_flat = []
        num_gts = []

        for pred_i, gt_i in zip(preds, gts):
            if isinstance(gt_i, str):
                gt_i = [gt_i]
            else:
                assert isinstance(gt_i, (list, tuple))

            preds_flat.extend([pred_i for _ in range(len(gt_i))])
            gts_flat.extend(list(gt_i))
            num_gts.append(len(gt_i))

        n_batches = int(math.ceil(len(preds_flat) / float(self.batch_size)))
        for i in tqdm(range(0, len(preds_flat), self.batch_size), total=n_batches, disable=not is_main_process()):
            pred_batch = preds_flat[i:i+self.batch_size]
            gt_batch = gts_flat[i:i+self.batch_size]

            user_msgs = [USER_MSG.format(gt, pred) for gt, pred in zip(gt_batch, pred_batch)]
            system_msgs = [SYSTEM_MSG for _ in range(len(user_msgs))]

            batch_outputs = self.model.predict_batch(system_msgs, user_msgs)

            for j, output in enumerate(batch_outputs):
                try:
                    score = float(output)
                except Exception as exc:
                    print(f"Output is not a float: {output}. Assigning zero score.")
                    print(f"Pred: {pred_batch[j]}\nGT: {gt_batch[j]}\n")
                    score = 0.0

                if score < 0 or score > 5:
                    print(f"Score is outside expected range: {score}")
                    score = min(5, max(0, score))

                scores_flat.append(score)

        scores_flat = torch.tensor(scores_flat, dtype=torch.float32)
        scores_split = torch.split(scores_flat, num_gts)
        if self.pooling == "max":
            scores_pooled = [x.max().item() for x in scores_split]
        elif self.pooling == "avg":
            scores_pooled = [x.mean().item() for x in scores_split]
        else:
            raise ValueError("Should not be here")

        assert len(scores_pooled) == len(preds)

        return scores_pooled