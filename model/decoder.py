from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange


class DecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, d_model: int, d_kv: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, kdim=d_kv, vdim=d_kv, device=device, dtype=dtype)

class DecoderBlock(nn.TransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)


class PromptLayer(nn.TransformerDecoderLayer):

    def __init__(self, d_model: int, d_kv: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, kdim=d_kv, vdim=d_kv, device=device, dtype=dtype)

class PromptDecoder(nn.TransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)


class FTNDecoder(torch.nn.Module):
    def __init__(self, in_dims: list[int], dim: int, dropout: float = 0.1) -> None:
        """in_dims: number of channel for stage 1, 2, 3 and 4"""
        super().__init__()

        self.linear_stage_2 = nn.Linear(in_features=in_dims[2], out_features=in_dims[1])
        self.linear_stage_3 = nn.Linear(in_features=in_dims[3], out_features=in_dims[2])

        self.linear2_stage_1 = nn.Linear(in_features=in_dims[0], out_features=dim)
        self.linear2_stage_2 = nn.Linear(in_features=in_dims[1], out_features=dim)
        self.linear2_stage_3 = nn.Linear(in_features=in_dims[2], out_features=dim)
        self.linear2_stage_4 = nn.Linear(in_features=in_dims[3], out_features=dim)

        self.attention_stage_2 = nn.ModuleList([
            FTNBlock(SRTransformerDecoder(d_model=dim, nhead=8, sr_ratio=2, dropout=dropout, batch_first=True))
            for _ in range(1)
        ])
        self.attention_stage_3 = nn.ModuleList([
            FTNBlock(SRTransformerDecoder(d_model=dim, nhead=8, sr_ratio=2, dropout=dropout, batch_first=True))
            for _ in range(2)
        ])
        self.attention_stage_4 = nn.ModuleList([
            FTNBlock(SRTransformerDecoder(d_model=dim, nhead=8, sr_ratio=2, dropout=dropout, batch_first=True))
            for _ in range(3)
        ])

    def forward(self, visual: Tensor, textual: Tensor):
        H = [int(t.shape[1] ** (1/2)) for t in visual]

        top_down_4 = visual[3]

        top_down_3 = rearrange(top_down_4, "b (h w) c -> b c h w", h=H[3])
        top_down_3 = F.interpolate(input=top_down_3, mode="bilinear", scale_factor=2)
        top_down_3 = self.linear_stage_3(rearrange(top_down_3, "b c h w -> b (h w) c", h=H[2]))

        top_down_2 = rearrange(top_down_3, "b (h w) c -> b c h w", h=H[2])
        top_down_2 = F.interpolate(input=top_down_2, mode="bilinear", scale_factor=2)
        top_down_2 = self.linear_stage_2(rearrange(top_down_2, "b c h w -> b (h w) c", h=H[1]))

        top_down_1 = visual[0]

        top_down_4 = self.linear2_stage_4(top_down_4)
        top_down_3 = self.linear2_stage_3(top_down_3)
        top_down_2 = self.linear2_stage_2(top_down_2)
        top_down_1 = self.linear2_stage_1(top_down_1)

        for mod in self.attention_stage_4:
            top_down_4 = mod(tgt=top_down_4, memory=textual)

        for mod in self.attention_stage_3:
            top_down_3 = mod(tgt=top_down_3, memory=textual)

        for mod in self.attention_stage_2:
            top_down_2 = mod(tgt=top_down_2, memory=textual)

        out = [top_down_1] + [top_down_2] + [top_down_3] + [top_down_4]
        summed = torch.stack(out, dim=0).sum(dim=0)

        return summed

class FTNBlock(nn.Module):

    def __init__(self, attention_block: nn.Module, upsample: int = 2) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.upsample = upsample

    def forward(self, tgt: Tensor, memory: Tensor):
        H = int(tgt.shape[1] ** (1/2))

        x = self.attention_block(tgt=tgt, memory=memory)
        x = rearrange(x, "b (h w) c -> b c h w", h=H)
        x = F.interpolate(input=x, mode="bilinear", scale_factor=self.upsample)
        x = rearrange(x, "b c h w -> b (h w) c", h=H*self.upsample)
        
        return x

class SRTransformerDecoder(nn.TransformerDecoderLayer):
    def __init__(self, d_model: int, nhead: int, sr_ratio: int = 1, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        self.sr_ratio = sr_ratio
        self.sr = torch.nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        self.norm = torch.nn.LayerNorm(d_model)

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        H = int(x.shape[1] ** (1/2))
        if self.sr_ratio > 1:
            reduced_x = self.sr(rearrange(x, "b (h w) c -> b c h w", h=H))
            reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c", h=int(H/self.sr_ratio))
            reduced_x = self.norm(reduced_x)
        else:
            reduced_x = x
        x = self.self_attn(x, reduced_x, reduced_x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # # multihead attention block
    # def _mha_block(self, x: Tensor, mem: Tensor,
    #                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
    #     H = int(x.shape[1] ** (1/2))
    #     if self.sr_ratio > 1:
    #         reduced_x = self.sr(rearrange(x, "b (h w) c -> b c h w", h=H))
    #         reduced_x = rearrange(reduced_x, "b c h w -> b (h w) c", h=int(H/self.sr_ratio))
    #         reduced_x = self.norm(reduced_x)
    #     else:
    #         reduced_x = x
    #     x = self.multihead_attn(x, reduced_x, reduced_x,
    #                             attn_mask=attn_mask,
    #                             key_padding_mask=key_padding_mask,
    #                             need_weights=False)[0]
    #     return self.dropout2(x)
