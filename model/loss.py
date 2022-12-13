from typing import Callable, Optional, Union

import numpy as np

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange


class AuxiliaryLoss(nn.CrossEntropyLoss):

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        B, H, W = target.shape
        input = F.interpolate(input=input, mode="bilinear", size=H)
        value = super().forward(input=input, target=target)
        return value

class NPairLoss(nn.Module):
    """Similar to CrossEntropyLoss"""

    def __init__(self, reduction: Union[Callable, None] = torch.mean) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: Tensor, x_pos: Tensor, x_neg: Tensor):
        
        pos = torch.matmul(x, x_pos.transpose(0,1))
        neg = torch.matmul(x, x_neg.transpose(0,1)).sum(-1, keepdim=True)
        res = (pos / (pos + neg)).sum(-1)
        if self.reduction:
            res = self.reduction(res)
        return res

class ContrastiveLoss(nn.Module):
    """Params: num_classes if we do not recompute the labels of the batch"""

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean', label_smoothing: float = 0) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, outputs: Tensor, labels: Tensor):
        # logits_per_text, logits_per_patch = outputs
        H = int(np.sqrt(outputs.shape[1]).item())

        # Define output
        out_textual = rearrange(outputs, "b (h w) c -> b h w c", h=H)
        out_visual = rearrange(outputs.transpose(-2, -1), "b c (h w) -> b c h w", h=H)
        
        # Define labels
        label_textual = F.one_hot(labels, num_classes=151).float()
        label_visual = labels

        # Define losses
        loss_textual = self.criterion(input=out_textual, target=label_textual)
        loss_visual = self.criterion(input=out_visual, target=label_visual)

        # print("Visual loss {}, Textual loss {}".format(loss_visual.item(), loss_textual.item()))

        return (loss_textual + loss_visual) / 2, loss_visual, loss_textual


