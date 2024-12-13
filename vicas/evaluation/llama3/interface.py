# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
import os.path as osp
from vicas.evaluation.llama3.generation import Dialog, Llama


class ModelWrapper:
    def __init__(self, ckpt_dir, max_seq_len=8192, max_batch_size=1):
        tokenizer_path = osp.join(ckpt_dir, "tokenizer.model")
        assert osp.exists(tokenizer_path), f"Tokenizer not found: {tokenizer_path}"

        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def predict(
            self, 
            system_msg: str, 
            user_msg: str,
            max_gen_len: Optional[int] = None,
            temperature: Optional[float] = 0.6,
            top_p: Optional[float] = 0.9
        ):
        dialog: Dialog = []
        if system_msg:
            dialog.append({
                "role": "system",
                "content": system_msg
            })

        assert user_msg
        dialog.append({
            "role": "user",
            "content": user_msg
        })
        
        results = self.generator.chat_completion(
            [dialog],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        return results[0]['generation']['content']

    def predict_batch(
        self,
        system_msgs: List[str],
        user_msgs: List[str], 
        max_gen_len: Optional[int] = None,
        temperature: Optional[float] = 0.6,
        top_p: Optional[float] = 0.9
    ):
        dialog_batch: List[Dialog] = []

        if system_msgs is not None:
            assert len(system_msgs) == len(user_msgs), f"len(system_msgs) = {len(system_msgs)}, len(user_msgs) = {len(user_msgs)}"

        for i in range(len(user_msgs)):
            dialog = []

            if system_msgs[i]:
                dialog.append({
                    "role": "system",
                    "content": system_msgs[i]
                })

            dialog.append({
                "role": "user",
                "content": user_msgs[i]
            })

            dialog_batch.append(dialog)
        
        results = self.generator.chat_completion(
            dialog_batch,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        return [results[i]['generation']['content'] for i in range(len(dialog_batch))]