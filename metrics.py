from typing import List, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchmetrics import JaccardIndex

from functools import wraps
import time
from tqdm import tqdm


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def prepare_for_label_metrics(outputs: Tensor, labels: Tensor, scale_factor: int = 4) -> tuple[List[Tensor], List[Tensor]]:
    """Prepare outputs and labels for metric compute. Outputs and labels are upsampled by scale factor then returned as a list of Tensor."""
    # outputs = F.interpolate(input=outputs, mode="bicubic", scale_factor=scale_factor)
    # labels = F.interpolate(input=labels.view(-1, 1, labels.shape[-1], labels.shape[-1]).float(), mode="nearest", scale_factor=scale_factor).squeeze().long()

    outputs_list = [x for x in outputs]
    labels_list = [x for x in labels]

    return outputs_list, labels_list

def prepare_for_gt_metrics(outputs: Tensor, gt_list: List[Tensor], sizes: Tensor) -> tuple[List[Tensor], List[Tensor]]:
    """Prepare outputs and ground truth labels for metric compute. Outputs are upsampled to their original size and returned as a list of Tensor."""
    outputs_list = [
        F.interpolate(input=output.unsqueeze(0), mode="bicubic", size=tuple(size)).squeeze()
        for output, size in zip(outputs, sizes)
    ]

    return outputs_list, gt_list


def segmentation_metrics(outputs: Tensor, labels: Tensor, gt_list: List[Tensor], sizes: Tensor, n_clas: int = 151, ignore_index: Optional[int] = 0) -> dict[str, float]:
    """Compute segmentation metrics"""
    do_labels, do_gt = True, True
    segm_metrics = {}
    
    if do_labels:
        label_outputs, label_labels = prepare_for_label_metrics(outputs=outputs, labels=labels, scale_factor=4)
        label_mIOU = compute_mIOU(pred=label_outputs, label=label_labels, n_cls=n_clas, ignore_index=ignore_index)
        segm_metrics = {**segm_metrics, **label_mIOU}
    if do_gt:
        gt_mIOU = compute_gt_mIOU(outputs=outputs, gt_list=gt_list, sizes=sizes, n_cls=n_clas, ignore_index=ignore_index)
        segm_metrics = {**segm_metrics, **gt_mIOU}
    
    return segm_metrics
    

def compute_gt_mIOU(outputs: Tensor, gt_list: List[Tensor], sizes: Tensor, n_cls: int = 151, ignore_index: Optional[int] = 0) -> dict[str, float]:
    """Compute mean IOU between list of predictions and list of ground truth labels"""
    softmax2D = nn.Softmax2d()
    jaccard = JaccardIndex(num_classes=n_cls, average="none")
    all_mIOU = []
    for i in tqdm(range(len(gt_list)), desc="Computing mIOU on gt", leave=False):
        pred = F.interpolate(input=outputs[i].unsqueeze(0), mode="bicubic", size=tuple(sizes[i])).squeeze()
        classes = gt_list[i].unique()
        iou = jaccard(softmax2D(pred).unsqueeze(0), gt_list[i].unsqueeze(0))
        
        if ignore_index is None:
            mIOU = iou[classes.long()].mean(dim=0, keepdim=True)
        else:
            mIOU = iou[classes[classes != ignore_index].long()].mean(dim=0, keepdim=True)
        
        all_mIOU.append(mIOU)

    mIOU = torch.concat(all_mIOU).mean().item()
    return dict(mIOU_gt=mIOU)


def compute_mIOU(pred: List[Tensor], label: list[Tensor], n_cls: int, ignore_index: Optional[int] = 0) -> dict[str, float]:
    """Compute mean IOU between list of predictions and list of labels"""
    softmax2D = nn.Softmax2d()
    jaccard = JaccardIndex(num_classes=n_cls, average="none")
    all_mIOU = []
    for i in tqdm(range(len(pred)), desc="Computing mIOU on label", leave=False):
        classes = label[i].unique()
        iou = jaccard(softmax2D(pred[i]).unsqueeze(0), label[i].unsqueeze(0))
        
        if ignore_index is None:
            mIOU = iou[classes.long()].mean(dim=0, keepdim=True)
        else:
            mIOU = iou[classes[classes != ignore_index].long()].mean(dim=0, keepdim=True)
        
        all_mIOU.append(mIOU)

    mIOU = torch.concat(all_mIOU).mean().item()
    return dict(mIOU_label=mIOU)





def compute_mIOU_tensor(pred: Tensor, label: Tensor, n_cls: int, ignore_index: Optional[int] = 0) -> float:
    """Compute mean IOU between predictions and labels"""
    softmax2D = nn.Softmax2d()
    jaccard = JaccardIndex(num_classes=n_cls, ignore_index=ignore_index)

    mIOU = jaccard(softmax2D(pred), label)
    
    return mIOU.item()


def original_size_interpolate(tensor: Tensor, ori_size: Tensor) -> List[Tensor]:
    """Interpolate each tensor to its corresponding sizes."""
    interpolated = [
        F.interpolate(input=t.unsqueeze(0), mode="bicubic", size=tuple(s)).squeeze()
        for t, s in zip(tensor, ori_size)
    ]
    return interpolated

def pad_and_concat(tensor_list: List[Tensor], ori_size: Tensor, pad: str = "max", value: int = 0) -> Tensor:
    """Pad and concat the tensor in tensor_list with their corresponding sizes."""
    if pad == "max":
        max_size = ori_size.max(0).values
    else:
        max_size = torch.LongTensor([1024, 1024])
    
    padded = [
        F.pad(n, pad=(0, max_size[1] - s[1], 0, max_size[0] - s[0]), mode="constant", value=value).unsqueeze(0)
        for n, s in zip(tensor_list, ori_size)
    ]

    return torch.cat(padded, dim=0)

def unpad(tensor: Tensor, size: Tensor) -> List[Tensor]:

    unpadded = [
        t[: s[0], :s[1]]
        for t, s in zip(tensor, size)
    ]
    return unpadded

def reshape_tensor(tensor: Tensor, ori_size: Tensor) -> Tensor:
    """Reshape each element in the tensor to its original corresponding size. Return a padded Tensor."""
    tensor_list = original_size_interpolate(tensor, ori_size)
    tensor = pad_and_concat(tensor_list, ori_size, pad="max", value=0)

    return tensor
