import PIL
from pathlib import Path
from tqdm import tqdm
import wandb

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchmetrics import JaccardIndex

from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

from dataset.dataset import SegmentationDataset, ClassDataset
from dataset.collator import Collator, FullClassCollator
from model.model import BaseModel

from utils import original_size_interpolate, pad_and_concat

torch.manual_seed(0)

DATA_NAME = "ade20k"
PATCH_SIZE = 16
IMG_SIZE = 224
LABEL_SIZE = (IMG_SIZE // PATCH_SIZE) * 4

MAX_EPOCH = 50
MAX_STEPS = 100
BATCH_SIZE = 128
LR = 1e-4

LOG_STEPS = 100
EVAL_STEPS = 100
RUN_NAME = "ade20k_block3_ignore0_128"
#WANBD_ARGS = dict(project="segm", name=RUN_NAME, mode="disabled")
WANBD_ARGS = dict(project="segm", name=RUN_NAME)

def train():

    train_data = ClassDataset(name=DATA_NAME, split="training")
    eval_data = ClassDataset(name=DATA_NAME, split="validation")

    img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=IMG_SIZE, crop_size=IMG_SIZE)
    label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=LABEL_SIZE, crop_size=LABEL_SIZE)
    collate = FullClassCollator(img_transform=img_transform, label_transform=label_transform, return_tensors="pt", padding=True)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, collate_fn=collate)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, collate_fn=collate, shuffle=False)

    model = BaseModel(patch=PATCH_SIZE, in_size=IMG_SIZE, out_size=LABEL_SIZE).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    max_steps = max(MAX_STEPS, MAX_EPOCH*len(train_loader))
    progress = tqdm(range(max_steps), desc="Training")
    wandb.init(**WANBD_ARGS)
    wandb.watch(model, log_freq=LOG_STEPS)
    wandb.define_metric("train/step")
    wandb.define_metric("*", step_metric="train/step")

    stop_training = False
    metrics = {}
    all_train_loss = None
    train_step = 0
    model.train()

    while not stop_training:
        for data in train_loader:
            train_step += 1
            progress.update()
            inputs, mappings, originals = data
            inputs = {k:v.cuda() for k,v in inputs.items()}

            optimizer.zero_grad()

            _, _, outputs = model(inputs)

            loss = criterion(outputs, inputs["label"])
            loss.backward()
            optimizer.step()

            all_train_loss = torch.concat([all_train_loss, loss.cpu().detach().unsqueeze(0)]) if all_train_loss is not None else loss.cpu().detach().unsqueeze(0)
            
            if (train_step % EVAL_STEPS) == 0:
                model.eval()
                eval_loss, eval_outputs, gt_list, eval_size = evaluate(model, eval_loader, criterion)
                metrics = {**metrics, **dict(eval_loss=eval_loss)}
                progress.set_postfix(metrics)
                # metrics = compute_metrics(eval_outputs, gt_list, eval_size)
                # print(metrics)
                model.train()

            if (train_step % LOG_STEPS) == 0:
                train_loss = all_train_loss.mean().item()
                epoch = round(train_step / len(train_loader), 4)

                metrics = {**metrics, **dict(train_epoch=epoch, train_loss=train_loss)}

                wandb.log({**{"train/step": train_step}, **{"/".join(k.split("_")): v for k,v in metrics.items()}})
                progress.set_postfix(metrics)
                
                all_train_loss = None
                Path("./checkpoints/{}/".format(RUN_NAME)).mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), "./checkpoints/{}/step-{}.pt".format(RUN_NAME, train_step))
                Path("./checkpoints/{}/step-{}.pt".format(RUN_NAME, train_step - LOG_STEPS)).unlink(missing_ok=True)


            if (train_step == max_steps):
                stop_training = True
                print("Training is over")
                break

            

def evaluate(model, eval_loader, criterion):
    eval_progress = tqdm(range(len(eval_loader)), desc="Evaluation", leave=False)

    all_eval_out = None
    all_eval_gt = None
    all_eval_size = None
    all_eval_loss = None

    
    for eval_step, data in enumerate(eval_loader):
        eval_progress.update()
        inputs, mappings, originals = data
        inputs = {k:v.cuda() for k,v in inputs.items()}

        with torch.no_grad():
            _, _, outputs = model(inputs)
            loss = criterion(outputs, inputs["label"])

        all_eval_out = torch.concat([all_eval_out, outputs]) if all_eval_out is not None else outputs
        all_eval_gt = all_eval_gt + originals["label"] if all_eval_gt is not None else originals["label"]
        all_eval_size = torch.concat([all_eval_size, inputs["size"]]) if all_eval_size is not None else inputs["size"]
        all_eval_loss = torch.concat([all_eval_loss, loss.unsqueeze(0)]) if all_eval_loss is not None else loss.unsqueeze(0)

    all_eval_out = all_eval_out.cpu()
    all_eval_size = all_eval_size.cpu()

    return all_eval_loss.mean().item(), all_eval_out, all_eval_gt, all_eval_size


def compute_metrics(pred, label, size):
    print(pred.shape)
    print([x.shape for x in label])
    print(size.shape)
    pred = original_size_interpolate(pred, size)
    # pred = pad_and_concat(pred, size, "max", 0)       # out of CPU memory
    # label = pad_and_concat(label, size, "max", 0)     # out of CPU memory
    
    sfm2D = torch.nn.Softmax2d()
    jaccard = JaccardIndex(num_classes=151, ignore_index=0)
    tot_mIOU = []
    for i in tqdm(range(len(pred))):
        mI0U = jaccard(sfm2D(pred[i]).unsqueeze(0), label[i].unsqueeze(0))
        tot_mIOU.append(mI0U)
    return dict(mIOU=sum(tot_mIOU)/len(tot_mIOU))

def pad(t, size):
    # label padding (pad on the right and bottom)
    label = [
        F.pad(label, pad=(0, 1024 - s[1], 0, 1024 - s[0]), mode="constant", value=0).unsqueeze(0)
        for label, s in zip(t, size)
    ]
    label = torch.cat(label, dim=0)

if __name__ == "__main__":
    train()