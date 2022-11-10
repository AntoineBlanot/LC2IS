import PIL
from pathlib import Path
from tqdm import tqdm
import wandb

import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

from data.dataset import SegmentationDataset, ClassDataset
from data.collator import Collator, FullClassCollator
from data.utils import save_img
from lc2is.model import BaseModel

from metrics import prepare_for_gt_metrics, prepare_for_label_metrics, compute_mIOU

from utils import apply_color

torch.manual_seed(0)

DATA_NAME = "ade20k"
PATCH_SIZE = 16
IMG_SIZE = 512
LABEL_SIZE = (IMG_SIZE // PATCH_SIZE) * 4

BATCH_SIZE = 16
DATA_SIZE = BATCH_SIZE * 5
LR = 1e-3
DROPOUT = 0.1
WEIGHT_DECAY = 1e-5
MIXED_PRECISION = True

MAX_EPOCH = 0
MAX_STEPS = 100
LOG_STEPS = 10
EVAL_STEPS = 10
SAVE_STEPS = 1000

DEVICE = "cuda"
RUN_NAME = "512_lr-3"
# WANBD_ARGS = dict(project="segm", name=RUN_NAME, mode="disabled")
WANBD_ARGS = dict(project="segm", name=RUN_NAME)

def train():

    train_data = ClassDataset(name=DATA_NAME, split="training")
    eval_data = ClassDataset(name=DATA_NAME, split="validation", size=DATA_SIZE)

    img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=IMG_SIZE, crop_size=IMG_SIZE)
    label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=LABEL_SIZE, crop_size=LABEL_SIZE)
    collate = FullClassCollator(img_transform=img_transform, label_transform=label_transform, return_tensors="pt", padding=True)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, collate_fn=collate)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False)

    model = BaseModel(patch=PATCH_SIZE, in_size=IMG_SIZE, out_size=LABEL_SIZE, dropout=DROPOUT, device=DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    max_steps = max(MAX_STEPS, MAX_EPOCH*len(train_loader))
    progress = tqdm(range(max_steps), desc="Training")

    wandb.init(**WANBD_ARGS)
    wandb.watch(model, log_freq=LOG_STEPS)
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")

    stop_training = False
    metrics = {}
    all_train_loss = []
    train_step = 0
    if MIXED_PRECISION:
        scaler = torch.cuda.amp.GradScaler()
    model.train()

    while not stop_training:
        for data in train_loader:
            train_step += 1
            inputs, mappings, originals = data
            inputs = {k: v.to(DEVICE) if k not in ["size"] else v for k,v in inputs.items()}

            optimizer.zero_grad()

            if MIXED_PRECISION:
                with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                    _, _, outputs = model(inputs)
                    loss = criterion(outputs, inputs["label"])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, _, outputs = model(inputs)
                loss = criterion(outputs, inputs["label"])
                loss.backward()
                optimizer.step()

            progress.update()

            all_train_loss.append(loss.item())
            
            if (train_step % EVAL_STEPS) == 0:
                model.eval()

                eval_loss, eval_outputs, eval_labels, eval_sizes, eval_gt = evaluate(model, eval_loader, criterion)
                metric_outputs, metric_labels = prepare_for_label_metrics(eval_outputs, eval_labels, scale_factor=4)
                eval_mIOU = compute_mIOU(metric_outputs, metric_labels, n_cls=151, ignore_index=0)
                
                metrics = {**metrics, **dict(eval_loss=eval_loss, eval_mIOU=eval_mIOU)}
                progress.set_postfix(metrics)

                model.train()

            if (train_step % LOG_STEPS) == 0:
                train_loss = np.array(all_train_loss).mean()
                epoch = round(train_step / len(train_loader), 4)

                metrics = {**metrics, **dict(train_epoch=epoch, train_loss=train_loss)}
                progress.set_postfix(metrics)
                wandb.log({**{"train/step": train_step}, **{"/".join(k.split("_")): v for k,v in metrics.items()}})
                
                all_train_loss = []

            if (train_step % SAVE_STEPS) == 0:
                Path("./checkpoints/{}/".format(RUN_NAME)).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), "./checkpoints/{}/step-{}.pt".format(RUN_NAME, train_step))
                Path("./checkpoints/{}/step-{}.pt".format(RUN_NAME, train_step - LOG_STEPS)).unlink(missing_ok=True)

            if (train_step == max_steps):
                stop_training = True
                print("Training is over")
                break

    last_outputs, last_gt = prepare_for_gt_metrics(eval_outputs, eval_gt, eval_sizes)
    mIOU = compute_mIOU(last_outputs, last_gt, n_cls=151, ignore_index=0)
    print("GT mIOU {}".format(mIOU))

    gt = [x for x in last_gt]
    masks = [x.argmax(dim=0) for x in last_outputs]

    colors = torch.LongTensor([[1, 255, 0, 255], [56, 0, 255, 0]])
    color_masks = [apply_color(x, colors) for x in masks]
    color_gt = [apply_color(x, colors) for x in gt]

    for i in range(len(gt)):
        gt, m = color_gt[i], color_masks[i]
        Path("./test/{}/".format(i)).mkdir(parents=True, exist_ok=True)
        save_img("./test/{}/gt.png".format(i), img=gt.to(torch.uint8))
        save_img("./test/{}/mask.png".format(i), img=m.to(torch.uint8))


def evaluate(model, eval_loader, criterion):
    eval_progress = tqdm(range(len(eval_loader)), desc="Evaluation", leave=False)

    all_eval_loss = []
    all_eval_out, all_eval_labels, all_eval_size = None, None, None
    all_eval_gt = []

    with torch.no_grad():
        for data in eval_loader:
            inputs, mappings, originals = data
            inputs = {k: v.to(DEVICE) if k not in ["size"] else v for k,v in inputs.items()}

            _, _, outputs = model(inputs)
            loss = criterion(outputs, inputs["label"])
            eval_progress.update()

            all_eval_loss.append(loss.item())
            all_eval_out = torch.concat([all_eval_out, outputs.cpu()]) if all_eval_out is not None else outputs.cpu()
            all_eval_labels = torch.concat([all_eval_labels, inputs["label"].cpu()]) if all_eval_labels is not None else inputs["label"].cpu()
            all_eval_size = torch.concat([all_eval_size, inputs["size"].cpu()]) if all_eval_size is not None else inputs["size"].cpu()
            all_eval_gt = all_eval_gt + originals["label"]

    eval_loss = np.array(all_eval_loss).mean()

    return eval_loss, all_eval_out, all_eval_labels, all_eval_size, all_eval_gt


if __name__ == "__main__":
    train()