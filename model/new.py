import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

from model.encoder import SwinTransformer
from model.decoder import FTNDecoder

from model.DenseCLIP.segmentation.denseclip.models import CLIPTextContextEncoder, ContextDecoder
from model.DenseCLIP.segmentation.denseclip.untils import tokenize

class Base(nn.Module):
    
    def __init__(self, cfg: dict, class_names: list[str], pretrained="model/DenseCLIP/segmentation/pretrained/ViT-B-16.pt") -> None:
        super().__init__()
        self.text_encoder = CLIPTextContextEncoder(**cfg["text_encoder"])
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.init_weights(pretrained=pretrained)
        self.vision_encoder = SwinTransformer()
        context_decoder_cfg = {**cfg["context_decoder"], **dict(visual_dim=1024), **dict(dropout=0.0)}
        self.context_decoder = ContextDecoder(**context_decoder_cfg)
        # self.aux_linear = nn.Linear(in_features=1024, out_features=512)
        self.decoder = FTNDecoder(in_dims=[128, 256, 512, 1024], dim=512, dropout=0)

        self.texts = torch.cat([tokenize(c, context_length=cfg["context_length"]) for c in class_names])
        self.num_classes = len(self.texts)

        context_length = self.text_encoder.context_length - cfg["context_length"]
        self.contexts = nn.Parameter(torch.randn(1, context_length, 512))
        nn.init.trunc_normal_(self.contexts)
        self.gamma = nn.Parameter(torch.ones(cfg["text_dim"]) * 1e-4)

    def forward(self, inputs):
        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        
        x = self.vision_encoder(**vision_inputs)
        stages = x[:4]
        visual_context = stages[-1]     # last stage
        B, P, C = visual_context.shape
        # print(visual_context.shape)

        # (B, K, C)
        text_embeddings = self.text_encoder(self.texts.to("cuda"), self.contexts).expand(B, -1, -1)
        # update text_embeddings by visual_context!
        # (B, 1, C)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        # (B, K, C)
        text_embeddings = text_embeddings + self.gamma * text_diff
        # print(text_embeddings.shape)

        # compute low resolution score map
        # low_res_vision = self.aux_linear(visual_context)
        # low_res_vision = rearrange(low_res_vision, "b (h w) c -> b c h w", h=16)
        # B, K, C = text_embeddings.shape
        # B, C, H, W = low_res_vision.shape
        # low_res_vision = F.normalize(low_res_vision, dim=1, p=2)
        # low_res_text = F.normalize(text_embeddings, dim=2, p=2)
        # low_res_score = torch.einsum('bchw,bkc->bkhw', low_res_vision, low_res_text)
        # print(low_res_score.shape)

        # decoding
        x = self.decoder(visual=stages, textual=text_embeddings)
        x = rearrange(x, "b (h w) c -> b c h w", h=128)
        x = F.normalize(x, dim=1, p=2)      # B C H W
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)         # B K C
        out = torch.einsum('bchw,bkc->bkhw', x, text_embeddings)
        # print(out.shape)

        out = F.interpolate(input=out, mode="bilinear", scale_factor=4)
        low_res_score = None
        return low_res_score, out
