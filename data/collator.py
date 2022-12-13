from typing import Any

import torch
import torch.nn.functional as F


class ADE20KCollator():

    def __init__(self) -> None:
        pass

    def __call__(self, features: list[Any]) -> tuple:
        img_list, label_list, metas_list = [list(f) for f in zip(*features)]

        img = torch.cat(img_list, dim=0)
        label = torch.cat(label_list, dim=0)
        metas = metas_list

        inputs = dict(pixel_values=img, label=label)
        return inputs, metas
