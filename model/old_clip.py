from PIL import Image
import requests

import torch.nn.functional as F
from einops import rearrange

from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as T

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
H, W = image.size

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
print(inputs.pixel_values.shape)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

new_image = outputs.vision_model_output.last_hidden_state[:, 1:, :]
print(new_image.shape)
# new_image = new_image.view(1, 1, 224, 224)
new_image = rearrange(new_image, "b (h w) c -> b c h w", h=224//16)
print(new_image.shape)
# pil = T.ToPILImage()(new_image[0])
# pil = pil.resize((224, 224), resample=Image.BICUBIC)
# print(pil.size)
# # new = F.interpolate(input=new_image, mode="bilinear", scale_factor=16)
# # print(new.shape)