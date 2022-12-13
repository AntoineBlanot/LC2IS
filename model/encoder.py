import torch
from torch import Tensor, nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPTextModel, SwinModel

from einops import rearrange

VIT_PRETRAINED_SIZE = 224

class ImageEncoderCLIP(nn.Module):

    def __init__(self, in_size: int, patch_size: int = 16) -> None:
        super().__init__()
        self.in_size = in_size
        self.patch_size = patch_size

        if self.patch_size == 16:
            path = "openai/clip-vit-base-patch16"

        self.enc = CLIPVisionModel.from_pretrained(path)
        
        # If fine-tuning size is different from pre-trained size, we need to 2D Interpolate the pos embeddings
        if self.in_size != VIT_PRETRAINED_SIZE:
            H = W = self.in_size // self.patch_size
            self.enc.vision_model.embeddings.position_ids = torch.arange((H * W) + 1).unsqueeze(0)
            self.enc.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(self.pos_emebedding_interpolate(tgt_size=H))

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.enc(pixel_values=pixel_values).last_hidden_state[:, 1:, :]
    
    def pos_emebedding_interpolate(self, tgt_size: int, ignore_index: int = 1) -> Tensor:
        """2D Interpolate the pos embedding of ViT because it has only been pre-trained on 224*224 size images"""
        old_size = VIT_PRETRAINED_SIZE // self.patch_size
        # ignore first index embedding (CLS token)
        ignored_pos = self.enc.vision_model.embeddings.position_embedding.weight[:ignore_index, :]
        old_pos = self.enc.vision_model.embeddings.position_embedding.weight[ignore_index:, :].unsqueeze(0)
        old_pos = rearrange(old_pos, "b (h w) c -> b c h w", h=old_size)
        # 2D upsampling
        new_pos = F.interpolate(input=old_pos, mode="bicubic", size=tgt_size)
        # rearange for putting back in vector
        new_pos = rearrange(new_pos, "b c h w -> b (h w) c", h=tgt_size).squeeze()
        
        return torch.cat([ignored_pos, new_pos], dim=0)

    def hidden_size(self) -> int:
        return self.enc.config.hidden_size

class ImageEncoderCLIPFull(nn.Module):

    def __init__(self, in_size: int, patch_size: int = 16) -> None:
        super().__init__()
        self.in_size = in_size
        self.patch_size = patch_size

        if self.patch_size == 16:
            path = "openai/clip-vit-base-patch16"

        self.enc = CLIPVisionModel.from_pretrained(path)
        
        # If fine-tuning size is different from pre-trained size, we need to 2D Interpolate the pos embeddings
        if self.in_size != VIT_PRETRAINED_SIZE:
            H = W = self.in_size // self.patch_size
            self.enc.vision_model.embeddings.position_ids = torch.arange((H * W) + 1).unsqueeze(0)
            self.enc.vision_model.embeddings.position_embedding.weight = torch.nn.Parameter(self.pos_emebedding_interpolate(tgt_size=H))

    def forward(self, pixel_values: Tensor) -> Tensor:
        return self.enc(pixel_values=pixel_values).last_hidden_state
    
    def pos_emebedding_interpolate(self, tgt_size: int, ignore_index: int = 1) -> Tensor:
        """2D Interpolate the pos embedding of ViT because it has only been pre-trained on 224*224 size images"""
        old_size = VIT_PRETRAINED_SIZE // self.patch_size
        # ignore first index embedding (CLS token)
        ignored_pos = self.enc.vision_model.embeddings.position_embedding.weight[:ignore_index, :]
        old_pos = self.enc.vision_model.embeddings.position_embedding.weight[ignore_index:, :].unsqueeze(0)
        old_pos = rearrange(old_pos, "b (h w) c -> b c h w", h=old_size)
        # 2D upsampling
        new_pos = F.interpolate(input=old_pos, mode="bicubic", size=tgt_size)
        # rearange for putting back in vector
        new_pos = rearrange(new_pos, "b c h w -> b (h w) c", h=tgt_size).squeeze()
        
        return torch.cat([ignored_pos, new_pos], dim=0)

    def hidden_size(self) -> int:
        return self.enc.config.hidden_size

class TextEncoderCLIP(nn.Module):

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size

        if self.patch_size == 16:
            path = "openai/clip-vit-base-patch16"

        self.enc = CLIPTextModel.from_pretrained(path)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def hidden_size(self) -> int:
        return self.enc.config.hidden_size

class TextEncoderCLIPPooler(nn.Module):

    def __init__(self, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size

        if self.patch_size == 16:
            path = "openai/clip-vit-base-patch16"

        self.enc = CLIPTextModel.from_pretrained(path)

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        return self.enc(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    def hidden_size(self) -> int:
        return self.enc.config.hidden_size

class SwinTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # path = "microsoft/swin-base-patch4-window7-224-in22k"
        path = "microsoft/swin-small-patch4-window7-224"
        self.encoder = SwinModel.from_pretrained(path)

    def forward(self, pixel_values: Tensor):
        features = self.encoder(pixel_values=pixel_values, output_hidden_states=True)
        return features.hidden_states[:4]