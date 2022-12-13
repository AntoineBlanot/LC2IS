import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

from model.encoder import SwinTransformer
from model.hierarchical import HierarchicalSelfA, HierarchicalCrossA

from model.DenseCLIP.segmentation.denseclip.models import CLIPTextContextEncoder, ContextDecoder
from model.DenseCLIP.segmentation.denseclip.untils import tokenize

class BaseSelfA(nn.Module):
    
    def __init__(self, class_names: list[str], dec_dim: int = 512, dec_depth: list[int] = [1, 1, 1], nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        # in_dims = [128, 256, 512, 1024]
        in_dims = [96, 192, 384, 768]
        self.vision_encoder = SwinTransformer()
        self.vision_decoder = HierarchicalSelfA(in_dims=in_dims, depth=dec_depth, dim=dec_dim, nhead=nhead, dropout=dropout)
        self.classes = nn.Linear(in_features=dec_dim, out_features=len(class_names), bias=False)

    def forward(self, inputs: dict) -> dict:
        vision_inputs = {k: v for k,v in inputs.items() if k in ["pixel_values"]}
        
        # Visual encoding
        v = self.vision_encoder(**vision_inputs)
        B, P, C = v[-1].shape
        t = self.classes.weight.expand(B, -1, -1)
        B, K, C = t.shape

        # Visual decoding
        v = self.vision_decoder(visual=v)
        B, P, C = v.shape
        H = W = int(P ** (1/2))

        v = rearrange(v, "b (h w) c -> b c h w", h=H)
        B, C, H, W = v.shape

        # Score map
        v = F.normalize(v, dim=1, p=2)
        t = F.normalize(t, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', v, t)
        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        # Outputs
        res_dict = dict(outputs=score_map)

        return res_dict

class BaseCrossA(nn.Module):
    
    def __init__(self, class_names: list[str], dec_dim: int = 512, dec_depth: list[int] = [1, 1, 1], nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        # in_dims = [128, 256, 512, 1024]
        in_dims = [96, 192, 384, 768]
        self.vision_encoder = SwinTransformer()
        self.vision_decoder = HierarchicalCrossA(in_dims=in_dims, depth=dec_depth, dim=dec_dim, nhead=nhead, dropout=dropout)
        self.classes = nn.Linear(in_features=dec_dim, out_features=len(class_names), bias=False)

    def forward(self, inputs: dict) -> dict:
        vision_inputs = {k: v for k,v in inputs.items() if k in ["pixel_values"]}
        
        # Visual encoding
        v = self.vision_encoder(**vision_inputs)
        B, P, C = v[-1].shape
        t = self.classes.weight.expand(B, -1, -1)
        B, K, C = t.shape

        # Visual decoding
        v = self.vision_decoder(visual=v, textual=t)
        B, P, C = v.shape
        H = W = int(P ** (1/2))

        v = rearrange(v, "b (h w) c -> b c h w", h=H)
        B, C, H, W = v.shape

        # Score map
        v = F.normalize(v, dim=1, p=2)
        t = F.normalize(t, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', v, t)
        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        # Outputs
        res_dict = dict(outputs=score_map)

        return res_dict

class PromptSelfA(nn.Module):
    
    def __init__(self, cfg: dict, class_names: list[str], dec_dim: int = 512, dec_depth: list[int] = [1, 1, 1], nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        # in_dims = [128, 256, 512, 1024]
        in_dims = [96, 192, 384, 768]
        pretrained_text = "model/DenseCLIP/segmentation/pretrained/ViT-B-16.pt"
        self.text_encoder = CLIPTextContextEncoder(**cfg["text_encoder"])
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.init_weights(pretrained=pretrained_text)
        self.texts = torch.cat([tokenize(c, context_length=cfg["context_length"]) for c in class_names])
        self.num_classes = len(self.texts)
        context_length = self.text_encoder.context_length - cfg["context_length"]
        self.contexts = nn.Parameter(torch.randn(1, context_length, 512))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(cfg["text_dim"]) * 1e-4)

        self.vision_encoder = SwinTransformer()

        context_decoder_cfg = {**cfg["context_decoder"], **dict(visual_dim=in_dims[-2]), **dict(dropout=0.0)}
        self.context_decoder = ContextDecoder(**context_decoder_cfg)
        
        self.vision_decoder = HierarchicalSelfA(in_dims=in_dims, depth=dec_depth, dim=dec_dim, nhead=nhead, dropout=dropout)

    def forward(self, inputs: dict) -> dict:
        vision_inputs = {k: v for k,v in inputs.items() if k in ["pixel_values"]}
        
        # Visual encoding
        v = self.vision_encoder(**vision_inputs)
        B, P, C = v[2].shape
        visual_context = v[2]   # stage 3

        # Prompt Encoding + Decoding
        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to("cuda"), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        t = text_embeddings + self.gamma * text_diff
        B, K, C = t.shape

        # Visual decoding
        v = self.vision_decoder(visual=v)
        B, P, C = v.shape
        H = W = int(P ** (1/2))

        v = rearrange(v, "b (h w) c -> b c h w", h=H)
        B, C, H, W = v.shape
        
        # Score map
        v = F.normalize(v, dim=1, p=2)
        t = F.normalize(t, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', v, t)
        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        # Outputs
        res_dict = dict(outputs=score_map)

        return res_dict

class PromptCrossA(nn.Module):
    
    def __init__(self, cfg: dict, class_names: list[str], dec_dim: int = 512, dec_depth: list[int] = [1, 1, 1], nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        # in_dims = [128, 256, 512, 1024]
        in_dims = [96, 192, 384, 768]
        pretrained_text = "model/DenseCLIP/segmentation/pretrained/ViT-B-16.pt"
        self.text_encoder = CLIPTextContextEncoder(**cfg["text_encoder"])
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.init_weights(pretrained=pretrained_text)
        self.texts = torch.cat([tokenize(c, context_length=cfg["context_length"]) for c in class_names])
        self.num_classes = len(self.texts)
        context_length = self.text_encoder.context_length - cfg["context_length"]
        self.contexts = nn.Parameter(torch.randn(1, context_length, 512))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(cfg["text_dim"]) * 1e-4)

        self.vision_encoder = SwinTransformer()

        context_decoder_cfg = {**cfg["context_decoder"], **dict(visual_dim=in_dims[-2]), **dict(dropout=0.0)}
        self.context_decoder = ContextDecoder(**context_decoder_cfg)

        self.vision_decoder = HierarchicalCrossA(in_dims=in_dims, depth=dec_depth, dim=dec_dim, nhead=nhead, dropout=dropout)

    def forward(self, inputs: dict) -> dict:
        vision_inputs = {k: v for k,v in inputs.items() if k in ["pixel_values"]}
        
        # Visual encoding
        v = self.vision_encoder(**vision_inputs)
        B, P, C = v[2].shape
        visual_context = v[2]   # stage 3

        # Prompt Encoding + Decoding
        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to("cuda"), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        t = text_embeddings + self.gamma * text_diff
        B, K, C = t.shape

        # Visual decoding
        v = self.vision_decoder(visual=v, textual=t)
        B, P, C = v.shape
        H = W = int(P ** (1/2))

        v = rearrange(v, "b (h w) c -> b c h w", h=H)
        B, C, H, W = v.shape
        
        # Score map
        v = F.normalize(v, dim=1, p=2)
        t = F.normalize(t, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', v, t)
        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        # Outputs
        res_dict = dict(outputs=score_map)

        return res_dict

class PromptAuxSelfA(nn.Module):
    
    def __init__(self, cfg: dict, class_names: list[str], dec_dim: int = 512, dec_depth: list[int] = [1, 1, 1], nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        # in_dims = [128, 256, 512, 1024]
        in_dims = [96, 192, 384, 768]
        pretrained_text = "model/DenseCLIP/segmentation/pretrained/ViT-B-16.pt"
        self.text_encoder = CLIPTextContextEncoder(**cfg["text_encoder"])
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.init_weights(pretrained=pretrained_text)
        self.texts = torch.cat([tokenize(c, context_length=cfg["context_length"]) for c in class_names])
        self.num_classes = len(self.texts)
        context_length = self.text_encoder.context_length - cfg["context_length"]
        self.contexts = nn.Parameter(torch.randn(1, context_length, 512))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(cfg["text_dim"]) * 1e-4)

        self.vision_encoder = SwinTransformer()

        context_decoder_cfg = {**cfg["context_decoder"], **dict(visual_dim=in_dims[-2]), **dict(dropout=0.0)}
        self.context_decoder = ContextDecoder(**context_decoder_cfg)

        self.aux_linear = nn.Linear(in_features=in_dims[-2], out_features=512)

        self.vision_decoder = HierarchicalSelfA(in_dims=in_dims, depth=dec_depth, dim=dec_dim, nhead=nhead, dropout=dropout)

    def forward(self, inputs: dict) -> dict:
        vision_inputs = {k: v for k,v in inputs.items() if k in ["pixel_values"]}
        
        # Visual encoding
        v = self.vision_encoder(**vision_inputs)
        B, P, C = v[2].shape
        visual_context = v[2]

        # Prompt Encoding + Decoding
        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to("cuda"), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        t = text_embeddings + self.gamma * text_diff
        B, K, C = t.shape

        # Low Resolution score map
        v_low = v[2]        # stage 3
        v_low = self.aux_linear(v_low)
        B, P, C = v_low.shape

        v_low = rearrange(v_low, "b (h w) c -> b c h w", h=int(P ** (1/2)))
        B, C, H, W = v_low.shape

        v_low = F.normalize(v_low, dim=1, p=2)
        t_low = F.normalize(t, dim=2, p=2)
        low_score_map = torch.einsum('bchw,bkc->bkhw', v_low, t_low)

        # Visual decoding
        v = self.vision_decoder(visual=v)
        B, P, C = v.shape

        v = rearrange(v, "b (h w) c -> b c h w", h=int(P ** (1/2)))
        B, C, H, W = v.shape
        
        # Score map
        v = F.normalize(v, dim=1, p=2)
        t = F.normalize(t, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', v, t)
        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        # Outputs
        res_dict = dict(outputs=score_map, low_score_map=low_score_map)

        return res_dict

class PromptAuxCrossA(nn.Module):
    
    def __init__(self, cfg: dict, class_names: list[str], dec_dim: int = 512, dec_depth: list[int] = [1, 1, 1], nhead: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        # in_dims = [128, 256, 512, 1024]
        in_dims = [96, 192, 384, 768]
        pretrained_text = "model/DenseCLIP/segmentation/pretrained/ViT-B-16.pt"
        self.text_encoder = CLIPTextContextEncoder(**cfg["text_encoder"])
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.init_weights(pretrained=pretrained_text)
        self.texts = torch.cat([tokenize(c, context_length=cfg["context_length"]) for c in class_names])
        self.num_classes = len(self.texts)
        context_length = self.text_encoder.context_length - cfg["context_length"]
        self.contexts = nn.Parameter(torch.randn(1, context_length, 512))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(cfg["text_dim"]) * 1e-4)

        self.vision_encoder = SwinTransformer()

        context_decoder_cfg = {**cfg["context_decoder"], **dict(visual_dim=in_dims[-2]), **dict(dropout=0.0)}
        self.context_decoder = ContextDecoder(**context_decoder_cfg)

        self.aux_linear = nn.Linear(in_features=in_dims[-2], out_features=512)

        self.vision_decoder = HierarchicalCrossA(in_dims=in_dims, depth=dec_depth, dim=dec_dim, nhead=nhead, dropout=dropout)

    def forward(self, inputs: dict) -> dict:
        vision_inputs = {k: v for k,v in inputs.items() if k in ["pixel_values"]}
        
        # Visual encoding
        v = self.vision_encoder(**vision_inputs)
        B, P, C = v[2].shape
        visual_context = v[2]

        # Prompt Encoding + Decoding
        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to("cuda"), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        t = text_embeddings + self.gamma * text_diff
        B, K, C = t.shape

        # Low Resolution score map
        v_low = v[2]        # stage 3
        v_low = self.aux_linear(v_low)
        B, P, C = v_low.shape

        v_low = rearrange(v_low, "b (h w) c -> b c h w", h=int(P ** (1/2)))
        B, C, H, W = v_low.shape

        v_low = F.normalize(v_low, dim=1, p=2)
        t_low = F.normalize(t, dim=2, p=2)
        low_score_map = torch.einsum('bchw,bkc->bkhw', v_low, t_low)

        # Visual decoding
        v = self.vision_decoder(visual=v, textual=t)
        B, P, C = v.shape

        v = rearrange(v, "b (h w) c -> b c h w", h=int(P ** (1/2)))
        B, C, H, W = v.shape
        
        # Score map
        v = F.normalize(v, dim=1, p=2)
        t = F.normalize(t, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', v, t)
        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        # Outputs
        res_dict = dict(outputs=score_map, low_score_map=low_score_map)

        return res_dict
