import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

from model.encoder import ImageEncoderCLIP, TextEncoderCLIP, TextEncoderCLIPPooler
from model.decoder import DecoderLayer, DecoderBlock
from model.text_patch import TextToPatch


class BaseModelWithText(nn.Module):

    def __init__(self, patch_size: int = 16, in_size: int = 224, out_size: int = 224, dropout: float = 0, num_layers: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_size = in_size
        self.out_size = out_size

        self.vision_encoder = ImageEncoderCLIP(in_size=self.in_size, patch_size=self.patch_size)
        self.text_encoder = TextEncoderCLIP(patch_size=self.patch_size)
        self.class_prototypes = nn.Parameter(torch.load("./model/ade20k_prototypes.pt"), requires_grad=True)
        vision_decoder_layer = DecoderLayer(d_model=768, d_kv=512, nhead=8, dropout=dropout, batch_first=True, norm_first=True)
        self.vision_decoder = DecoderBlock(decoder_layer=vision_decoder_layer, num_layers=num_layers)
        self.pixel_patch = TextToPatch(out=512, img_in=self.vision_encoder.hidden_size(), text_in=self.text_encoder.hidden_size())
        
    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor]:

        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        text_inputs = {k:v for k,v in inputs.items() if k in ["input_ids", "attention_mask"]}

        enc_t = self.text_encoder(**text_inputs)
        # print("Textual encoder {} {}".format(enc_t.shape, enc_t.dtype))

        enc_v = self.vision_encoder(**vision_inputs)
        # print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        dec_v = self.vision_decoder(tgt=enc_v, memory=enc_t, memory_key_padding_mask=torch.where(text_inputs["attention_mask"] == 1, False, True))
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))

        H = self.in_size // self.patch_size
        dec_v = rearrange(dec_v, "b (h w) c -> b c h w", h=H) 
        dec_v = F.interpolate(input=dec_v, mode="bicubic", scale_factor=4)
        dec_v = rearrange(dec_v, "b c h w -> b (h w) c", h=self.out_size)
        # print("Vision decoder {} {}".format(dec_v.shape, dec_v.dtype))

        feature_t, feature_v = self.pixel_patch(img=dec_v, text=self.class_prototypes)
        # print("Pixel-Patch {} {}, {} {}".format(feature_t.shape, feature_t.dtype, feature_v.shape, feature_v.dtype))

        feature_mm = torch.matmul(feature_v, feature_t.transpose(1, 0))
        # print("MM feature {} {}".format(feature_mm.shape, feature_mm.dtype))

        feature_mm = rearrange(feature_mm, "b (h w) c -> b c h w", h=self.out_size)
        # print("MM feature {} {}".format(feature_mm.shape, feature_mm.dtype))

        return feature_t, feature_v, feature_mm


class ContrastiveModel(nn.Module):

    def __init__(self, patch_size: int = 16, in_size: int = 224, out_size: int = 224, dropout: float = 0, num_layers: int = 1) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_size = in_size
        self.out_size = out_size

        self.vision_encoder = ImageEncoderCLIP(in_size=self.in_size, patch_size=self.patch_size)
        self.text_encoder = TextEncoderCLIPPooler(patch_size=self.patch_size)
        self.pixel_patch = TextToPatch(out=512, img_in=self.vision_encoder.hidden_size(), text_in=self.text_encoder.hidden_size())
        self.temperature = nn.Parameter(data=torch.Tensor([0.07]), requires_grad=True)

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor]:
       
        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        text_inputs = {k:v for k,v in inputs.items() if k in ["input_ids", "attention_mask"]}

        enc_t = self.text_encoder(**text_inputs)
        print("Textual encoder {} {}".format(enc_t.shape, enc_t.dtype))

        enc_v = self.vision_encoder(**vision_inputs)
        print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        H = self.in_size // self.patch_size
        enc_v = rearrange(enc_v, "b (h w) c -> b c h w", h=H)
        enc_v = F.interpolate(input=enc_v, mode="bicubic", scale_factor=4)
        enc_v = rearrange(enc_v, "b c h w -> b (h w) c", h=self.out_size)
        print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        feature_t, feature_v = self.pixel_patch(img=enc_v, text=enc_t)
        print("Pixel-Patch {} {}, {} {}".format(feature_t.shape, feature_t.dtype, feature_v.shape, feature_v.dtype))

        feature_mm = torch.matmul(feature_v, feature_t.transpose(1, 0)) * torch.exp(self.temperature)
        print("MM feature {} {}".format(feature_mm.shape, feature_mm.dtype))

        return feature_t, feature_v, feature_mm
