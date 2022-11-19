import argparse
from pathlib import Path
import json

import PIL

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

from engine import Engine
from data.dataset import ADE20K_Dataset
from data.collator import JoinTextCollator
from model.model import BaseModelWithText
from metrics import segmentation_metrics

def get_args():
    parser = argparse.ArgumentParser()

    # Base arguments
    parser.add_argument("--name", type=str, required=True, help="Run name")
    parser.add_argument("--out_dir", type=str, help="Outputs directory")
    parser.add_argument("--seed", type=int, default=1024, help="Seed for reproducibility")

    # Data arguments
    parser.add_argument("--data_name", type=str, help="Dataset name")
    parser.add_argument("--data_size", type=int, help="Dataset size")

    # Training arguments
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--img_size", type=int, default=512, help="Input images size")
    parser.add_argument("--label_size", type=int, default=128, help="Input labels size")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate (initial)")
    parser.add_argument("--dropout", type=float, help="Dropout")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--device", type=str, help="Device that will be use")
    parser.add_argument("--fp16", type=bool, help="Whether to activate mixed precision")
    parser.add_argument("--max_epoch", type=int, default=1, help="Maximum train epoch")
    parser.add_argument("--max_steps", type=int, help="Maximum train steps")
    parser.add_argument("--log_step", type=int, help="Step at which log (at each epoch if not specified)")
    parser.add_argument("--eval_step", type=int, help="Step at which evaluate (at each epoch if not specified)")
    parser.add_argument("--save_step", type=int, help="Step at which save (at each epoch if not specified)")

    # Logging arguments
    parser.add_argument("--wandb_project", type=str, help="Name of WandB project")

    args = parser.parse_args()

    return args

args = get_args()

# Save config
Path(args.out_dir+args.name+"/").mkdir(parents=True, exist_ok=True)
with open(args.out_dir+args.name+"/config.json", "w") as fp:
    json.dump(vars(args), fp)


# Build dataset and data loaders
torch.manual_seed(args.seed)
train_data = ADE20K_Dataset(split="training", size=args.data_size)
eval_data = ADE20K_Dataset(split="training", size=args.data_size)
img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=args.img_size, crop_size=args.img_size)
label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=args.label_size, crop_size=args.label_size)
txt_transform = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
collate = JoinTextCollator(img_transform=img_transform, label_transform=label_transform, txt_transform=txt_transform, return_tensors="pt", padding=True)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, collate_fn=collate)
eval_loader = DataLoader(dataset=eval_data, batch_size=args.batch_size, collate_fn=collate, shuffle=False)


# Build model, optimizer, criterion and lr_scheduler
model = BaseModelWithText(patch_size=args.patch_size, in_size=args.img_size, out_size=args.label_size, dropout=args.dropout, num_layers=3)
optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()
lr_sceduler = None


# Define logger
logger = "wandb"
logger_args = dict(project=args.wandb_project, name=args.name)


# Define engine
trainer = Engine(
    name=args.name, out_dir=args.out_dir,
    model=model, optimizer=optimizer, criterion=criterion, lr_scheduler=lr_sceduler,
    device=args.device, fp16=args.fp16,
    train_loader=train_loader, eval_loader=eval_loader, compute_metrics=segmentation_metrics,
    max_epoch=args.max_epoch, max_steps=args.max_steps, eval_step=args.eval_step, log_step=args.log_step, save_step=args.save_step,
    logger=logger, logger_args=logger_args
)

# Training
train_metrics, save_dir = trainer.train()

print(train_metrics)
print("Model checkpoints saved at path: {}".format(save_dir))