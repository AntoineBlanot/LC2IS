import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

from model.encoder import ImageEncoderCLIP, ImageEncoderCLIPFull, TextEncoderCLIP, TextEncoderCLIPPooler, SwinTransformer
from model.decoder import DecoderLayer, DecoderBlock, PromptLayer, PromptDecoder, FTNDecoder
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
        # self.temperature = nn.Parameter(data=torch.Tensor([0.07]), requires_grad=True)

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor]:
       
        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        text_inputs = {k:v for k,v in inputs.items() if k in ["input_ids", "attention_mask"]}

        enc_t = self.text_encoder(**text_inputs)
        # print("Textual encoder {} {}".format(enc_t.shape, enc_t.dtype))

        enc_v = self.vision_encoder(**vision_inputs)
        # print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        H = self.in_size // self.patch_size
        enc_v = rearrange(enc_v, "b (h w) c -> b c h w", h=H)
        enc_v = F.interpolate(input=enc_v, mode="bicubic", scale_factor=4)
        enc_v = rearrange(enc_v, "b c h w -> b (h w) c", h=self.out_size)
        # print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        feature_t, feature_v = self.pixel_patch(img=enc_v, text=enc_t)
        # print("Pixel-Patch {} {}, {} {}".format(feature_t.shape, feature_t.dtype, feature_v.shape, feature_v.dtype))

        # logits = torch.matmul(feature_v, feature_t.transpose(1, 0)) * torch.exp(self.temperature)
        logits = torch.matmul(feature_v, feature_t.transpose(1, 0))
        # logits_per_patch = logits
        # logits_per_text = logits.transpose(-2, -1)
        # print("MM feature {} {}".format(logits.shape, logits.dtype))

        # logits_per_patch = rearrange(logits_per_patch, "b (h w) c -> b h w c", h=self.out_size)
        # logits_per_text = rearrange(logits_per_text, "b c (h w) -> b c h w", h=self.out_size)
        # print("Logits per patch {} {}".format(logits_per_patch.shape, logits_per_patch.dtype))
        # print("Logits per text {} {}".format(logits_per_text.shape, logits_per_text.dtype))

        return feature_t, feature_v, logits


class DenseClip(nn.Module):

    def __init__(self, patch_size, in_size, out_size) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.in_size = in_size
        self.out_size = out_size

        self.vision_encoder = ImageEncoderCLIPFull(in_size=self.in_size, patch_size=self.patch_size)
        self.text_encoder = TextEncoderCLIPPooler(patch_size=self.patch_size) #freeze
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_patch = TextToPatch(out=512, img_in=self.vision_encoder.hidden_size(), text_in=self.text_encoder.hidden_size())
        prompt_layer = PromptLayer(d_model=512, nhead=8)
        self.prompt_decoder = PromptDecoder(prompt_layer, num_layers=8)
        vision_decoder_layer = DecoderLayer(d_model=768, d_kv=512, nhead=8, batch_first=True, norm_first=True)
        self.vision_decoder = DecoderBlock(decoder_layer=vision_decoder_layer, num_layers=8)
        

    def forward(self, inputs: dict[str, Tensor]) -> tuple[Tensor]:

        B = inputs["pixel_values"].shape[0]
       
        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        text_inputs = {k:v for k,v in inputs.items() if k in ["input_ids", "attention_mask"]}

        enc_t = self.text_encoder(**text_inputs)
        # print("Textual encoder {} {}".format(enc_t.shape, enc_t.dtype))

        enc_v = self.vision_encoder(**vision_inputs)
        # print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        # H = self.in_size // self.patch_size
        # enc_v = rearrange(enc_v, "b (h w) c -> b c h w", h=H)
        # enc_v = F.interpolate(input=enc_v, mode="bicubic", scale_factor=4)
        # enc_v = rearrange(enc_v, "b c h w -> b (h w) c", h=self.out_size)
        # print("Vision encoder {} {}".format(enc_v.shape, enc_v.dtype))

        feature_t, feature_v = self.text_patch(img=enc_v, text=enc_t)
        feature_t = feature_t.expand(B, -1, -1)
        # print("Pixel-Patch {} {}, {} {}".format(feature_t.shape, feature_t.dtype, feature_v.shape, feature_v.dtype))

        v_context = self.prompt_decoder(tgt=feature_t, memory=feature_v)
        # print("Vision context {} {}".format(v_context.shape, v_context.dtype))

        text_embeddings = feature_t + 1e-5 * v_context
        # print("Text emebddings {} {}".format(v_context.shape, v_context.dtype))

        H = self.in_size // self.patch_size
        visual_embeddings = rearrange(feature_v[:, 1:, :], "b (h w) c -> b c h w", h=H)
        # visual_embeddings = F.interpolate(input=visual_embeddings, mode="bicubic", scale_factor=4)
        # print("Visual emebddings {} {}".format(visual_embeddings.shape, visual_embeddings.dtype))

        # B, C, H, W = visual_embeddings.shape
        B, K, C = text_embeddings.shape
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text = F.normalize(text_embeddings, dim=2, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text)
        # print("Score maps {} {}".format(score_map.shape, score_map.dtype))

        out = self.vision_decoder(tgt=enc_v, memory=text_embeddings)
        print("Vision decode {} {}".format(out.shape, out.dtype))

        # new score map here

        return None, score_map, out


class PromptFTN(torch.nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        self.textual_encoder = TextEncoderCLIPPooler(patch_size=16) #freeze
        for param in self.textual_encoder.parameters():
            param.requires_grad = False
        
        self.visual_encoder = SwinTransformer()
        self.prompt_decoder = PromptDecoder(PromptLayer(d_model=512, d_kv=1024, nhead=8, batch_first=True), num_layers=8)
        self.decoder = FTNDecoder(in_dims=[128, 256, 512, 1024], dim=512)

        # self.gamma = nn.Parameter(torch.ones(512) * 1e-4)

    def forward(self, inputs):
        B = inputs["pixel_values"].shape[0]
        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        text_inputs = {k:v for k,v in inputs.items() if k in ["input_ids", "attention_mask"]}
        
        text_embeddings = self.textual_encoder(**text_inputs).expand(B, -1, -1)
        features = self.visual_encoder(**vision_inputs)
        stages = features[:4]
        global_feature = stages[-1]     # last stage

        visual_context = self.prompt_decoder(tgt=text_embeddings, memory=global_feature)
        # text_embeddings = text_embeddings + self.gamma * visual_context
        text_embeddings = text_embeddings + 1e-4 * visual_context
        # print(global_feature.shape, text_embeddings.shape)
        
        visual_embeddings = self.decoder(visual=stages, textual=text_embeddings)
        # print(visual_embeddings.shape)

        visual_embeddings = rearrange(visual_embeddings, "b (h w) c -> b c h w", h=128)
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)      # B C H W
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)         # B K C
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_embeddings)
        # print(score_map.shape)

        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        return None, score_map
