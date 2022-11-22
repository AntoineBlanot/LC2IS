import torch
from torch import Tensor, nn

class TextToPatch(nn.Module):

    def __init__(self, img_in: int, text_in: int, out: int = 512) -> None:
        super().__init__()
        # img   (batch, patches, img_in)   -->     (batch, patches, out)
        # text  (classes, text_in)          -->    (classes, out)

        self.textual = nn.Linear(in_features=text_in, out_features=out)
        self.visual = nn.Linear(in_features=img_in, out_features=out)
    
    def forward(self, img: Tensor, text: Tensor) -> tuple[Tensor, Tensor]:

        t_feature = self.textual(text)
        v_feature = self.visual(img)

        return t_feature, v_feature