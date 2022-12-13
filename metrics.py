from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchmetrics.classification import MulticlassJaccardIndex

from tqdm import tqdm


def meanIOU(outputs: Tensor, labels: Tensor, ignore_index: Optional[int] = 0) -> dict:
    """Compute mean IOU"""
    B, C, H, W = outputs.shape
    jaccard = MulticlassJaccardIndex(num_classes=C, average="none")
    all_mIOU = []

    for i in tqdm(range(B), desc="Computing mIOU", leave=False):
        pred, label = outputs[i].unsqueeze(0), labels[i].unsqueeze(0)
        classes = label.unique().long()
        if ignore_index is not None:
            classes = classes[classes != ignore_index]
        iou = jaccard(pred, label)
        mIOU = iou[classes].mean(dim=0, keepdim=True)
        all_mIOU.append(mIOU)
        
    mIOU = torch.stack(all_mIOU).mean().item()
    res_dict = dict(mIOU=mIOU)
    return res_dict
