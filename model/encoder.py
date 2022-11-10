import torch
from transformers import CLIPVisionModel, CLIPTextModel

from einops import rearrange

class ImageEncoderCLIP(torch.nn.Module):

    def __init__(self, patch: int = 16) -> None:
        super().__init__()
        self.patch = patch

        if self.patch == 16:
            path = "openai/clip-vit-base-patch16"

        self.enc = CLIPVisionModel.from_pretrained(path)

        self.enc.vision_model.embeddings.position_ids = torch.arange(1025).unsqueeze(0)
        self.enc.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(self.pos_emebedding_interpolate(tgt_size=32))


    def forward(self, inputs):
        return self.enc(inputs).last_hidden_state[:, 1:, :]
    
    def pos_emebedding_interpolate(self, tgt_size: int, ignore_index: int = 1):
        """2D Interpolate the pos embedding of ViT because it has only been pre-trained on 224*224 size images"""
        # ignore first index embedding (CLS token)
        new_pos = self.enc.vision_model.embeddings.position_embedding.weight[ignore_index:, :].unsqueeze(0)
        new_pos = rearrange(new_pos, "b (h w) c -> b c h w", h=14)
        # rearange for Upsampling
        new_pos = torch.nn.Upsample(size=tgt_size, mode="bicubic")(new_pos)
        new_pos = rearrange(new_pos, "b c h w -> b (h w) c", h=tgt_size).squeeze()
        # rearange for putting back in vector
        return torch.cat([self.enc.vision_model.embeddings.position_embedding.weight[:ignore_index, :], new_pos], dim=0)

class TextEncoderCLIP(torch.nn.Module):

    def __init__(self, patch: int = 16) -> None:
        super().__init__()
        self.patch = patch

        if self.patch == 16:
            path = "openai/clip-vit-base-patch16"

        self.enc = CLIPTextModel.from_pretrained(path)

    def forward(self, input_ids, attention_mask = None):
        return self.enc(input_ids, attention_mask).pooler_output