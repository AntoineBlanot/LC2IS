from transformers import AutoFeatureExtractor, SwinModel
import torch
import torch.nn.functional as F
from einops import rearrange

from model.encoder import TextEncoderCLIPPooler
from model.decoder import DecoderLayer, PromptDecoder

class BaseFTN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
        self.decoder = Decoder()
        self.classif = torch.nn.Linear(in_features=512, out_features=151)

    def forward(self, inputs):
        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        
        features = self.backbone(**vision_inputs, output_hidden_states=True)
        stages = list(features.hidden_states)[:-1]
        out = self.decoder(stages)
        cls = self.classif(out)

        reshaped = rearrange(cls, "b (h w) c -> b c h w", h=128)
        reshaped = F.interpolate(input=reshaped, mode="bilinear", scale_factor=4)

        return None, reshaped

class PromptFTN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = TextEncoderCLIPPooler(patch_size=16) #freeze
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.backbone = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
        prompt_layer = DecoderLayer(d_model=512, d_kv=512, nhead=8, batch_first=True)
        self.prompt_decoder = PromptDecoder(prompt_layer, num_layers=8)
        self.decoder = Decoder()

    def forward(self, inputs):
        B = inputs["pixel_values"].shape[0]
        vision_inputs = {k:v for k,v in inputs.items() if k in ["pixel_values"]}
        text_inputs = {k:v for k,v in inputs.items() if k in ["input_ids", "attention_mask"]}
        # print(vision_inputs["pixel_values"].shape)
        
        classes = self.text_encoder(**text_inputs)
        features = self.backbone(**vision_inputs, output_hidden_states=True)
        stages = list(features.hidden_states)[:-1]
        visual_embeddings = self.decoder(stages)
        # print(visual_embeddings.shape)

        v_context = self.prompt_decoder(tgt=classes.expand(B, -1, -1), memory=visual_embeddings)
        text_embeddings = classes + 1e-5 * v_context
        # print(visual_embeddings.shape, text_embeddings.shape)

        visual_embeddings = rearrange(visual_embeddings, "b (h w) c -> b c h w", h=128)
        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_embeddings)
        # print(score_map.shape)

        score_map = F.interpolate(input=score_map, mode="bilinear", scale_factor=4)

        return None, score_map


class Decoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim_in = [128, 256, 512, 1024]      # [96, 192, 384, 768]
        dim_out = [256, 512, 1024, 1024]    # [192, 384, 768, 768]
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(in_features=dim_in[0], out_features=dim_out[0]),
            torch.nn.Linear(in_features=dim_in[1], out_features=dim_out[1]),
            torch.nn.Linear(in_features=dim_in[2], out_features=dim_out[2]),
            torch.nn.Linear(in_features=dim_in[3], out_features=dim_out[3])
        ])
        # self.linears = torch.nn.ModuleList([
        #     torch.nn.Linear(in_features=dim_out[0], out_features=dim_in[0]),
        #     torch.nn.Linear(in_features=dim_out[1], out_features=dim_in[1]),
        #     torch.nn.Linear(in_features=dim_out[2], out_features=dim_in[2]),
        #     torch.nn.Linear(in_features=dim_out[3], out_features=dim_in[3])
        # ])
        self.attentions = torch.nn.ModuleList([
            Transformer(repeat=1, upsample=False, sr_ratio=1, dim=512, nhead=1),
            Transformer(repeat=1, upsample=True, sr_ratio=2, dim=512, nhead=8),
            Transformer(repeat=2, upsample=True, sr_ratio=2, dim=512, nhead=8),
            Transformer(repeat=3, upsample=True, sr_ratio=2, dim=512, nhead=8),
        ])
        self.linears2 = torch.nn.ModuleList([
            torch.nn.Linear(in_features=dim_out[0], out_features=512),
            torch.nn.Linear(in_features=dim_out[1], out_features=512),
            torch.nn.Linear(in_features=dim_out[2], out_features=512),
            torch.nn.Linear(in_features=dim_out[3], out_features=512)
        ])
        # self.linears2 = torch.nn.ModuleList([
        #     torch.nn.Linear(in_features=dim_in[0], out_features=512),
        #     torch.nn.Linear(in_features=dim_in[1], out_features=512),
        #     torch.nn.Linear(in_features=dim_in[2], out_features=512),
        #     torch.nn.Linear(in_features=dim_in[3], out_features=512)
        # ])

    def forward(self, x):   
        H = [128, 64, 32, 16]
        # print([i.shape for i in x])
        n_stages = len(x)
        add = []
        for i in range(n_stages-1, -1, -1):
            if i == n_stages-1 or i == 0:
                add += [torch.Tensor()]
            else:
                r = rearrange(x[i+1], "b (h w) c -> b c h w", h=H[i+1])
                r = F.interpolate(input=r, mode="bilinear", scale_factor=2)
                add += [rearrange(r, "b c h w -> b (h w) c", h=H[i])]
                # try this (line 94)
                # add += [self.linears[i](rearrange(r, "b c h w -> b (h w) c", h=H[i]))]
        add = [x for x in reversed(add)]
        # print([i.shape for i in add])
        # ok until here, now has to linear the base so that channel dim match the add
        
        # out = [x[i] for i in range(n_stages)]
        out = [self.linears[i](x[i]) for i in range(n_stages)]
        out = [out[i] + add[i] if i in [1, 2] else out[i] for i in range(n_stages)]
        # print([i.shape for i in out])

        end = [self.attentions[i](self.linears2[i](out[i]), h=H[i]) for i in range(1, n_stages)]
        end = [self.linears2[0](out[0])] + end
        # print([i.shape for i in end])
        return torch.stack(end, dim=0).sum(dim=0)

class Transformer(torch.nn.Module):
    def __init__(self, repeat, sr_ratio, dim, upsample=True, nhead=8) -> None:
        super().__init__()
        self.trans = torch.nn.ModuleList([
            torch.nn.TransformerDecoder(torch.nn.TransformerDecoderLayer(d_model=dim, nhead=nhead, batch_first=True), num_layers=1)
            for _ in range(repeat)
        ])
        self.upsample = upsample
        self.sr_ratio = sr_ratio
        self.sr = torch.nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
        self.norm = torch.nn.LayerNorm(512)

    def forward(self, x, h):
        if self.sr_ratio > 1:
            memory = self.sr(rearrange(x, "b (h w) c -> b c h w", h=h))
            memory = rearrange(memory, "b c h w -> b (h w) c", h=int(h/self.sr_ratio))
            memory = self.norm(memory)
        else:
            memory = x

        for layer in self.trans:
            x = layer(tgt=x, memory=memory)
            if self.upsample:
                x = rearrange(x, "b (h w) c -> b c h w", h=h)
                x = F.interpolate(input=x, mode="bilinear", scale_factor=2)
                x = rearrange(x, "b c h w -> b (h w) c", h=h*2)
        return x
    
# dataset = load_dataset("huggingface/cats-image")
# image = dataset["test"]["image"][0]

# feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
# decoder = Decoder()

# inputs = feature_extractor(image, return_tensors="pt")
# inputs = dict(pixel_values=torch.Tensor(1, 3, 512, 512))

# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=True)
#     stages = list(outputs.hidden_states)[:-1]
#     outputs = decoder(stages)

# # print([x.shape for x in outputs.hidden_states])
# # print([x.shape for x in outputs.reshaped_hidden_states])

# print(outputs.shape)
# print(rearrange(outputs, "b (h w) c -> b c h w", h=128).shape)