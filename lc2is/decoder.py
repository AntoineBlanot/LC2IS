from typing import Callable, Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class TransformerDecoder(torch.nn.Module):

    def __init__(self, img_in: int = 768, text_in: int = 512, dim_feedforward: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()

        # Self-Attention
        self.norm1 = nn.LayerNorm(img_in)
        self.smha = nn.MultiheadAttention(embed_dim=img_in, num_heads=8, batch_first=True, dropout=dropout)
        # Cross-Attention
        self.norm2 = nn.LayerNorm(img_in)
        self.cmha = nn.MultiheadAttention(embed_dim=img_in, kdim=text_in, vdim=text_in, num_heads=8, batch_first=True, dropout=dropout)
        
        # MLP
        self.norm3 = nn.LayerNorm(img_in)
        self.linear1 = nn.Linear(in_features=img_in, out_features=dim_feedforward)
        self.linear2 = nn.Linear(in_features=dim_feedforward, out_features=img_in)

    def forward(self, img, text):
        """img_attn_mask should be always None since all images are reshaped to fix sized by CLIPProcessor (no padding)"""
        # Self-Attention
        img = self.norm1(img)
        y_sa, _ = self.smha(query=img, key=img, value=img)
        y_sa = y_sa.softmax(dim=-1)

        # Cross-Attention
        y_sa = self.norm2(y_sa)
        y_ca, _ = self.cmha(query=y_sa, key=text, value=text)

        # MLP
        y_ca = self.norm3(y_ca)
        y_mlp = self.linear2(self.linear1(y_ca))

        # Missing pos encoding
        # Missing feed forward
        # Missing dropout
        # Missing other ? ...
        return y_mlp

class DecoderLayer(nn.TransformerDecoderLayer):

    def __init__(self, d_model: int, d_kv: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0, activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, kdim=d_kv, vdim=d_kv, **factory_kwargs)

class DecoderBlock(nn.TransformerDecoder):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(decoder_layer, num_layers, norm)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return super().forward(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        