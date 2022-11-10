from typing import Callable, Union
import torch

class NPairLoss(torch.nn.Module):
    """Similar to CrossEntropyLoss"""

    def __init__(self, reduction: Union[Callable, None] = torch.mean) -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.Tensor, x_pos: torch.Tensor, x_neg: torch.Tensor):
        
        pos = torch.matmul(x, x_pos.transpose(0,1))
        neg = torch.matmul(x, x_neg.transpose(0,1)).sum(-1, keepdim=True)
        res = (pos / (pos + neg)).sum(-1)
        if self.reduction:
            res = self.reduction(res)
        return res

