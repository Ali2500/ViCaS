from typing import List, Tuple, Any, Dict
from dataclasses import dataclass, asdict
from tabulate import tabulate
import numpy as np

CROWD_ID = -1


@dataclass
class VideoCaptionPhrase:
    ids: List[int]
    phrase: str
    start_idx: int
    end_idx: int
    start_idx_raw: int
    end_idx_raw: int

    def to_dict(self):
        return {
            "ids": self.ids,
            "phrase": self.phrase,
            "pos_start": self.start_idx,
            "pos_end": self.end_idx 
        }


@dataclass
class VideoCaption:
    raw: str
    parsed: str
    objects: List[VideoCaptionPhrase]

    def __str__(self):
        lines = []
        lines.append(f"Raw:    {self.raw}")
        lines.append(f"Parsed: {self.parsed}")

        table_rows = [["IDs", "Position", "Phrase"]]
        for obj in self.objects:
            ids = ','.join([str(x) for x in obj.ids])
            table_rows.append([ids, f"[{obj.start_idx}-{obj.end_idx}]", obj.phrase])
            # lines.append(f"-> {ids}: [{obj.start_idx}-{obj.end_idx}]: {obj.phrase}")

        lines.append(tabulate(table_rows, headers='firstrow', tablefmt="pretty"))
        return "\n".join(lines)
    
    def to_dict(self):
        return {
            "raw": self.raw,
            "parsed": self.parsed,
            "objects": [obj.to_dict() for obj in self.objects]
        }

    def object_ids(self, exclude_crowd=True):
        all_ids = sum([obj.ids for obj in self.objects], [])
        ret = sorted(list(set(all_ids)))
        if CROWD_ID in ret and exclude_crowd:
            ret.remove(CROWD_ID)

        return ret


def parse_caption(caption: str) -> VideoCaption:
    obj_phrase_delimiters = ['[', ']']
    mask_delimiters = ['<', '>']
    parsed = ""
    objs = []
    curr_start = -1
    curr_end = -1
    curr_start_raw = -1
    curr_end_raw = -1
    curr_phrase = ""
    curr_ids = []
    state = 'txt'  # 'txt', 'in_phrase', 'in_id'

    caption = caption.replace("？", "?")
    caption = caption.replace("，", ",")

    def get_past_context(idx, length=25):
        context = f"Position: {idx}. History: " + "\""
        if idx > length:
            start = idx - length
            context = context + "..."
        else:
            start = 0

        return context + caption[start:idx+1] + "\""

    i = 0
    while i < len(caption):
        try:
            char = caption[i]
            if state == 'txt':
                if char == obj_phrase_delimiters[0]:
                    state = 'in_phrase'
                elif char in mask_delimiters + [obj_phrase_delimiters[1]]:
                    raise ValueError(f"Should not be here (1). Encountered: {char}. {get_past_context(i)}")
                else:
                    parsed = parsed + char

            elif state == 'in_phrase':
                if char == obj_phrase_delimiters[1]:
                    curr_end = len(parsed) - 1
                    curr_end_raw = i
                    assert caption[i+1] == mask_delimiters[0], f"Unexpected char: {caption[i+1]}. {get_past_context(i+1)}"
                    i += 2

                    prefix_found = False
                    for redundant_prefix in ("mask_", "mask"):
                        if caption[i:i+len(redundant_prefix)] == redundant_prefix:
                            prefix_found = True
                            break

                    assert prefix_found, f"Unexpected sequence: {caption[i:i+10]}. {get_past_context(i)}"

                    state = 'in_id'
                    i = i + len(redundant_prefix)
                    continue
                elif char in mask_delimiters + [obj_phrase_delimiters[0]]:
                    raise ValueError(f"Should not be here (2). Encountered: {char}. {get_past_context(i)}")
                else:
                    parsed = parsed + char
                    if curr_start == -1:
                        curr_start = len(parsed) - 1
                        curr_start_raw = i

                    curr_phrase = curr_phrase + char

            elif state == 'in_id':
                if char == mask_delimiters[1]:
                    assert curr_ids, f"No IDs were parsed in the current ID sequence string"
                    objs.append(VideoCaptionPhrase(curr_ids.copy(), curr_phrase, curr_start, curr_end, curr_start_raw, curr_end_raw))
                    curr_start = -1
                    curr_end = -1
                    curr_phrase = ""
                    curr_ids.clear()
                    state = 'txt'

                elif char in (",", " "):
                    i += 1
                    continue
                elif char == "?":
                    curr_ids.append(CROWD_ID)
                else:
                    curr_ids.append(int(char))

            i += 1

        except Exception as err:
            print(f"Original Error: {err}")
            raise ValueError(f"Error parsing caption at index {i}. Context: {get_past_context(i)}")

    return VideoCaption(caption, parsed, objs) 
