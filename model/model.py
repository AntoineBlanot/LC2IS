from typing import Tuple

import torch
from torch import Tensor, nn

from einops import rearrange

from model.encoder import ImageEncoderCLIP, TextEncoderCLIP
from model.decoder import DecoderLayer, DecoderBlock
from model.text_patch import TextToPatch


class BaseModel(nn.Module):

    def __init__(self, patch: int = 16, in_size: int = 224, out_size: int = 1024, dropout: float = 0, device: str = "cuda") -> None:
        super().__init__()
        self.patch = patch
        self.in_size = in_size
        self.out_size = out_size

        self.vision_encoder = ImageEncoderCLIP(patch=self.patch)
        # self.text_encoder = TextEncoderCLIP(patch=self.patch)
        self.text_proto = torch.load("./lc2is/ade20k_prototypes.pt").to(device)
        # self.vision_decoder = TransformerDecoder(img_in=768, text_in=512)
        vision_decoder_layer = DecoderLayer(d_model=768, d_kv=512, nhead=8, dropout=dropout, batch_first=True, norm_first=True)
        self.vision_decoder = DecoderBlock(decoder_layer=vision_decoder_layer, num_layers=1)
        self.pixel_patch = TextToPatch(out=512, img_in=768, text_in=512)
        
    def forward(self, inputs) -> Tuple[Tensor]:
        img = inputs["pixel_values"]
        # text = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]

        #n_cls = text.shape[0]
        n_cls = len(self.text_proto)
        batch_size = img.shape[0]

        # print("n_classes {}".format(n_cls))

        enc_v = self.vision_encoder(img)
        # print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        # enc_t = self.text_encoder(input_ids=text, attention_mask=attention_mask)
        enc_t = self.text_proto
        # print("Textual encoder {} {}".format(enc_t.shape, enc_t.dtype))

        expanded_text = enc_t.expand(batch_size, -1, -1)
        dec_v = self.vision_decoder(img=enc_v, text=expanded_text)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))

        H = self.in_size // self.patch
        dec_v = rearrange(dec_v, "b (h w) c -> b c h w", h=H)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))
        up4 = torch.nn.Upsample(scale_factor=4, mode="bicubic")
        dec_v = up4(dec_v)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))
        dec_v = rearrange(dec_v, "b c h w -> b (h w) c", h=self.out_size)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))

        # view imilar to rearrange(dec_v, "b h w c -> b (h w) c")

        feature_v, feature_t = self.pixel_patch(img=dec_v, text=enc_t)
        # print("Pixel-Patch {} {}, {} {}".format(feature_v.shape, feature_v.dtype, feature_t.shape, feature_t.dtype))

        feature_mm = torch.matmul(feature_t, feature_v.transpose(1, -1))
        # print("MM feature {} {}".format(feature_mm.shape, feature_mm.dtype))

        # feature_mm = feature_mm.view(-1, n_cls, self.out_size, self.out_size)
        feature_mm = rearrange(feature_mm, "b c (h w) -> b c h w", h=self.out_size)
        # print("MM feature {} {}".format(feature_mm.shape, feature_mm.dtype))

        return feature_t, feature_v, feature_mm

class BaseModelWithText(nn.Module):

    def __init__(self, patch: int = 16, in_size: int = 224, out_size: int = 1024, dropout: float = 0) -> None:
        super().__init__()
        self.patch = patch
        self.in_size = in_size
        self.out_size = out_size

        self.vision_encoder = ImageEncoderCLIP(patch=self.patch)
        self.text_encoder = TextEncoderCLIP(patch=self.patch)
        self.class_prototypes = nn.Parameter(torch.load("./model/ade20k_prototypes.pt"), requires_grad=True)
        vision_decoder_layer = DecoderLayer(d_model=768, d_kv=512, nhead=8, dropout=dropout, batch_first=True, norm_first=True)
        self.vision_decoder = DecoderBlock(decoder_layer=vision_decoder_layer, num_layers=1)
        self.pixel_patch = TextToPatch(out=512, img_in=768, text_in=512)
        
    def forward(self, inputs) -> Tuple[Tensor]:

        enc_v = self.vision_encoder(pixel_values=inputs["pixel_values"])
        # print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        enc_t = self.text_encoder(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        # print("Textual encoder {} {}".format(enc_t.shape, enc_t.dtype))

        dec_v = self.vision_decoder(tgt=enc_v, memory=enc_t, memory_key_padding_mask=torch.where(inputs["attention_mask"] == 1, False, True))
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))

        H = self.in_size // self.patch
        dec_v = rearrange(dec_v, "b (h w) c -> b c h w", h=H)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))

        up4 = torch.nn.Upsample(scale_factor=4, mode="bicubic")
        dec_v = up4(dec_v)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))
        dec_v = rearrange(dec_v, "b c h w -> b (h w) c", h=self.out_size)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))

        # view imilar to rearrange(dec_v, "b h w c -> b (h w) c")

        feature_v, feature_t = self.pixel_patch(img=dec_v, text=self.class_prototypes)
        # print("Pixel-Patch {} {}, {} {}".format(feature_v.shape, feature_v.dtype, feature_t.shape, feature_t.dtype))

        feature_mm = torch.matmul(feature_t, feature_v.transpose(1, -1))
        # print("MM feature {} {}".format(feature_mm.shape, feature_mm.dtype))

        # feature_mm = feature_mm.view(-1, n_cls, self.out_size, self.out_size)
        feature_mm = rearrange(feature_mm, "b c (h w) -> b c h w", h=self.out_size)
        # print("MM feature {} {}".format(feature_mm.shape, feature_mm.dtype))

        return feature_t, feature_v, feature_mm
