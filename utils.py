from typing import List
import torch
import torch.nn.functional as F


def count_params(model: torch.nn.Module, trainable: bool = False):
    """Return the number of parameters (all or trainlable only) in millions"""
    if trainable:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        params = sum(p.numel() for p in model.parameters())

    return params / 1e6

def generate_masks(preds: torch.Tensor, sizes: torch.Tensor) -> List[torch.Tensor]:
    
    upsampled = [
        F.interpolate(input=pred.unsqueeze(0), mode="bicubic", size=tuple(size)).squeeze()
        for pred,size in zip(preds, sizes)
    ]
    masks = [x.argmax(dim=0) for x in upsampled]
    return masks

def apply_color(t: torch.Tensor, colors: torch.Tensor):
    # masks --> [H, W]
    # colors -> [[class_id, dim1, dim2, dim3], ...]
    color_mask = (t.flatten() == colors[:, :1])
    dim1 = (color_mask * colors[:, 1:2]).sum(0, keepdim=True).reshape(t.shape).unsqueeze(0)
    dim2 = (color_mask * colors[:, 2:3]).sum(0, keepdim=True).reshape(t.shape).unsqueeze(0)
    dim3 = (color_mask * colors[:, 3:]).sum(0, keepdim=True).reshape(t.shape).unsqueeze(0)
    dims = torch.concat([dim1, dim2, dim3])
    expand_mask = (1 - color_mask.sum(0)).reshape(t.shape).expand(3, -1, -1)
    
    return (expand_mask * t) + dims

def original_size_interpolate(tensor: torch.Tensor, ori_size: torch.Tensor) -> List[torch.Tensor]:
    """Interpolate each tensor to its corresponding sizes."""
    interpolated = [
        F.interpolate(input=t.unsqueeze(0), mode="bicubic", size=tuple(s)).squeeze()
        for t, s in zip(tensor, ori_size)
    ]
    return interpolated

def pad_and_concat(tensor_list: List[torch.Tensor], ori_size: torch.Tensor, pad: str = "max", value: int = 0) -> torch.Tensor:
    """Pad and concat the tensor in tensor_list with their corresponding sizes."""
    if pad == "max":
        max_size = ori_size.max(0).values
    else:
        max_size = torch.LongTensor([1024, 1024])
    
    padded = [
        F.pad(n, pad=(0, max_size[1] - s[1], 0, max_size[0] - s[0]), mode="constant", value=value).unsqueeze(0)
        for n, s in zip(tensor_list, ori_size)
    ]
    concat_padded = torch.cat(padded, dim=0)
    return concat_padded