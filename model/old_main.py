from PIL import Image
import requests
import torch

import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPProcessor, CLIPModel, CLIPTextModel
from decoder import TransformerDecoder
from text_patch import TextToPatch

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
decoder = TransformerDecoder(img_dim=768, text_dim=512)
text_patch = TextToPatch(multi_modal_emb=512, img_dim=768, text_dim=512)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url2 = "http://images.cocodataset.org/val2017/000000454661.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)

# 1: cat, 2: dog, 3: car, 4: person, 5: tree
class_id = [[1, 2], [3, 4, 5]]
class_txt = [["cat, dog"], ["car, person, tree"]]
classes = sum(class_txt, [])

inputs = processor(text=classes, images=[image, image2], return_tensors="pt", padding=True)
print({k: v.shape for k,v in inputs.items()})
text_inputs = processor(text=["cat", "dog", "car", "person", "tree"], return_tensors="pt", padding=True)
print({k: v.shape for k,v in text_inputs.items()})

# CLIP: Image + group text
clip_outputs = clip(**inputs)
img_emb = clip_outputs.vision_model_output.last_hidden_state
text_emb = clip_outputs.text_model_output.last_hidden_state
print("CLIPModel vision {}, text {}".format(img_emb.shape, text_emb.shape))

# CLIP: Text only (classes)
clip_text_outputs = text_encoder(**text_inputs).pooler_output
print("CLIPText output {}".format(clip_text_outputs.shape))

# Attention
attn_output = decoder(img=img_emb, text=clip_text_outputs.expand(2, clip_text_outputs.shape[0], clip_text_outputs.shape[1]))
print("Image decoder output {}".format(attn_output.shape))

# 1D image to 2D image
visual_feature = attn_output[:, 1:, :]       # exclude cls token of ViT
reshaped_visual_feature = rearrange(visual_feature, "b (h w) c -> b c h w", h=224//16) #img_size/patch_size
print("Reshaped image {}".format(reshaped_visual_feature.shape))

## contrastive loss here !!
v_feature, t_feature = text_patch(attn_output, clip_text_outputs)
print("TextToPatch visual {}, textual {}".format(v_feature.shape, t_feature.shape))
mat_mul = torch.matmul(v_feature, torch.transpose(t_feature, 0, 1))
print("Matrix Mul {}".format(mat_mul.shape))


# to obtain final segmentation mask
sizes = [image.size, image2.size]
upsampled = [
    F.interpolate(input=x.unsqueeze(0), mode="bicubic", size=s)
    for x,s in zip(rearrange(mat_mul[:, 1:, :], "b (h w) c -> b c h w", h=224//16), sizes)
]
upsampled = [
    F.pad(x, pad=(0, 1024 - s[1], 0, 1024 - s[0]), mode="constant", value=0)
    for x, s in zip(upsampled, sizes)
]
final = torch.concat(upsampled, dim=0)
print(final.shape)

