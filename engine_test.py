import PIL

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

from engine import Engine
from data.dataset import ClassDataset
from data.collator import TextCollator
from lc2is.model import BaseModelWithText
from metrics import compute_gt_mIOU

name = "overfit"
model = BaseModelWithText(patch=16, in_size=512, out_size=128, dropout=0)
optimizer = optim.Adam(params=model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()


train_data = ClassDataset(name="ade20k", split="training", size=16*4)
eval_data = ClassDataset(name="ade20k", split="training", size=16*4)
img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=512, crop_size=512)
label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=128, crop_size=128)
txt_transform = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
collate = TextCollator(img_transform=img_transform, label_transform=label_transform, txt_transform=txt_transform, return_tensors="pt", padding=True)
train_loader = DataLoader(dataset=train_data, batch_size=16, collate_fn=collate)
eval_loader = DataLoader(dataset=eval_data, batch_size=16, collate_fn=collate, shuffle=False)

trainer = Engine(
    name=name,
    model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=None,
    device="cuda", fp16=False,
    train_loader=train_loader, eval_loader=eval_loader, compute_metrics=compute_gt_mIOU,
    max_epoch=100, max_steps=-1, eval_step="epoch", log_step="epoch", save_step=-100, save_dir="./", logger=None
)

metrics, save_path = trainer.train()

print(metrics)
print(save_path)
