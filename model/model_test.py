from typing import List
import PIL
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torchmetrics import JaccardIndex

from data.dataset import SegmentationDataset, ClassDataset
from data.collator import Collator, FullClassCollator
from data.utils import save_img

from torch.utils.data import DataLoader
from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

from model.model import Model

def generate_masks(preds: torch.Tensor, sizes: torch.Tensor, id_mapping: torch.Tensor, do_max: bool = True) -> List[torch.Tensor]:
    masks = [
        F.interpolate(input=pred.unsqueeze(0), mode="bicubic", size=tuple(size)).squeeze()
        for pred, size in zip(preds, sizes)
    ]
    if do_max:
        masks = [x.argmax(dim=0) for x in masks]
        masks = [putback_ids(m, id_mapping) for m in masks]
    return masks

def apply_color(t: torch.Tensor, colors: torch.Tensor):
    # masks --> [H, W]
    # colors -> [[class_id, dim1, dim2, dim3], ...]
    color_mask = (t.flatten() == colors[:, :1])
    dim1 = (color_mask * colors[:, 1:2]).sum(0, keepdim=True).reshape(t.shape).unsqueeze(0)
    dim2 = (color_mask * colors[:, 2:3]).sum(0, keepdim=True).reshape(t.shape).unsqueeze(0)
    dim3 = (color_mask * colors[:, 3:]).sum(0, keepdim=True).reshape(t.shape).unsqueeze(0)
    dims = torch.concat([dim1, dim2, dim3])
    expand_mask = (1 - color_mask.sum(0)).reshape(t.shape).expand(3, -1, -1)
    
    return (expand_mask * t) + dims

def putback_ids(x, id_mapping):
    # print(x.unique(), len(x.unique()), x.shape)
    flattened_x = x.flatten()
    mask = (flattened_x == id_mapping[:, 1:])
    flattened_x = (1 - mask.sum(dim=0)) * flattened_x + (mask * id_mapping[:,:1]).sum(dim=0)
    x = flattened_x.reshape(x.shape)
    # print(x.unique(), len(x.unique()), x.shape)
    return x

# data = SegmentationDataset(name="ade20k", split="training")
data = ClassDataset(name="ade20k", split="training")

LABEL_SIZE = (224 // 16) * 4
img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16")
label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=LABEL_SIZE, crop_size=LABEL_SIZE)
# txt_transform = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
# collate = Collator(img_transform=img_transform, txt_transform=txt_transform, label_transform=label_transform, return_tensors="pt", padding=True)
collate = FullClassCollator(img_transform=img_transform, label_transform=label_transform, return_tensors="pt", padding=True)
loader = DataLoader(dataset=data, batch_size=10, collate_fn=collate)

index = 10

i = iter(loader)
inputs, mappings, originals = next(i)
print({k: v.shape for k,v in inputs.items()})
print({k: len(v) for k, v in mappings.items()})
print({k: len(v) for k, v in originals.items()})

model = Model(patch=16, in_size=224, out_size=LABEL_SIZE)
model.text_proto = model.text_proto.cpu()
criterion = CrossEntropyLoss(ignore_index=0)

label = inputs["label"]
size = inputs["size"]

out = model(inputs)
print("Model output {}, label {}".format(out.shape, label.shape))

loss = criterion(out, label)
print("Loss value {}".format(loss))

# eval
new_out = [
    F.interpolate(input=pred.unsqueeze(0), mode="bicubic", size=tuple(size)).squeeze()
    for pred, size in zip(out, size)
]
print("Interpolation at original size {}".format([x.shape for x in new_out]))

maxes = size.max(0).values
concat_out = [
    F.pad(n, pad=(0, maxes[1] - s[1], 0, maxes[0] - s[0]), mode="constant", value=0).unsqueeze(0)
    for n, s in zip(new_out, size)
]
concat_out = torch.cat(concat_out, dim=0)
print("Concatenation with max padding {}".format(concat_out.shape))

sfm2D = torch.nn.Softmax2d()
jaccard = JaccardIndex(num_classes=151, ignore_index=0)
mIOU = jaccard(sfm2D(out), label)
print("Mean IOU {}".format(mIOU))

gt_labels = originals["label"]
masks = generate_masks(out, size, mappings["old_new_id_mapping"])
colors = torch.LongTensor([[1, 255, 0, 255], [56, 0, 255, 0]])

color_masks = [apply_color(x, colors) for x in masks]
color_labels = [apply_color(x, colors) for x in gt_labels]

for i in range(len(gt_labels)):
    gt, m = color_labels[i], color_masks[i]
    Path("./test/{}/".format(i)).mkdir(parents=True, exist_ok=True)
    save_img("./test/{}/label.png".format(i), img=gt.to(torch.uint8))
    save_img("./test/{}/mask.png".format(i), img=m.to(torch.uint8))
