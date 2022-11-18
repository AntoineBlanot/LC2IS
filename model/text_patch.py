import torch

class TextToPatch(torch.nn.Module):

    def __init__(self, out: int = 512, img_in: int = 768, text_in: int = 512) -> None:
        super().__init__()
        # img   (batch, patches, img_in)   -->     (batch, patches, out)
        # text  (classes, text_in)          -->    (classes, out)

        self.visual = torch.nn.Linear(in_features=img_in, out_features=out)
        self.textual = torch.nn.Linear(in_features=text_in, out_features=out)

    
    def forward(self, img: torch.Tensor, text: torch.Tensor):

        v_feature = self.visual(img)
        t_feature = self.textual(text)

        return v_feature, t_feature