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
from dataset.utils import save_img
from model.model import BaseModel

from metrics import compute_mIOU_list, prepare_for_label_metrics, prepare_for_gt_metrics

torch.manual_seed(0)

DATA_NAME = "ade20k"
PATCH_SIZE = 16
IMG_SIZE = 512
LABEL_SIZE = (IMG_SIZE // PATCH_SIZE) * 4

MAX_EPOCH = 300
MAX_STEPS = 1
BATCH_SIZE = 8
LR = 1e-4
DROPOUT = 0

LOG_STEPS = 1
EVAL_STEPS = 5*50
SAVE_STEPS = 5*50
DEVICE = "cuda"
RUN_NAME = "512_data8*5_all_nodropout"
STEP = 1500
PATH = Path("checkpoints/{}/step-{}.pt".format(RUN_NAME, STEP))

from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

# @timeit
# def eval_tensor_gt():

#     eval_data = ClassDataset(name=DATA_NAME, split="training", size=BATCH_SIZE*5*5)

#     img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=IMG_SIZE, crop_size=IMG_SIZE)
#     label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=LABEL_SIZE, crop_size=LABEL_SIZE)
#     collate = FullClassCollator(img_transform=img_transform, label_transform=label_transform, return_tensors="pt", padding=True)
#     eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, collate_fn=collate)

#     model = BaseModel(patch=PATCH_SIZE, in_size=IMG_SIZE, out_size=LABEL_SIZE, dropout=DROPOUT, device=DEVICE).to(DEVICE)
#     model.load_state_dict(torch.load(PATH))
#     model.eval()
#     criterion = nn.CrossEntropyLoss()

#     eval_loss, eval_output, eval_label, eval_size, eval_gt = evaluate(model, eval_loader, criterion)
#     print("Eval loss {}".format(eval_loss))
#     print("Previous shapes {}\n{}".format(eval_output.shape, [x.shape for x in eval_gt]))

#     eval_output = reshape_tensor(eval_output, eval_size).cpu()
#     eval_gt = pad_and_concat(eval_gt, eval_size, pad="max", value=0).cpu()
#     print("New shapes {} {}".format(eval_output.shape, eval_gt.shape))

#     metric = compute_mIOU(pred=eval_output, label=eval_gt, n_cls=151, ignore_index=0)
#     print("Metric {}".format(metric))

#     eval_masks = eval_output.argmax(dim=1)
#     print("Masks shape {}".format(eval_masks.shape))

#     eval_masks = unpad(eval_masks, eval_size)
#     eval_gt = unpad(eval_gt, eval_size)
#     print("Original shapes {}\n{}".format([x.shape for x in eval_masks], [x.shape for x in eval_gt]))

#     colors = torch.LongTensor([[1, 255, 0, 255], [56, 0, 255, 0]])
#     color_masks = [apply_color(x, colors) for x in eval_masks]
#     color_labels = [apply_color(x, colors) for x in eval_gt]

#     for i in range(len(eval_gt)):
#         gt, m = color_labels[i], color_masks[i]
#         Path("./test/{}/".format(i)).mkdir(parents=True, exist_ok=True)
#         save_img("./test/{}/label.png".format(i), img=gt.to(torch.uint8))
#         save_img("./test/{}/mask.png".format(i), img=m.to(torch.uint8))

# @timeit
# def eval_tensor_label():

#     eval_data = ClassDataset(name=DATA_NAME, split="training", size=BATCH_SIZE*5*5)

#     img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=IMG_SIZE, crop_size=IMG_SIZE)
#     label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=LABEL_SIZE, crop_size=LABEL_SIZE)
#     collate = FullClassCollator(img_transform=img_transform, label_transform=label_transform, return_tensors="pt", padding=True)
#     eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, collate_fn=collate)

#     model = BaseModel(patch=PATCH_SIZE, in_size=IMG_SIZE, out_size=LABEL_SIZE, dropout=DROPOUT, device=DEVICE).to(DEVICE)
#     model.load_state_dict(torch.load(PATH))
#     model.eval()
#     criterion = nn.CrossEntropyLoss()

#     eval_loss, eval_output, eval_label, eval_size, eval_gt = evaluate(model, eval_loader, criterion)
#     print("Eval loss {}".format(eval_loss))
#     print("Previous shapes {}\n{}".format(eval_output.shape, [x.shape for x in eval_gt]))

#     up4 = torch.nn.Upsample(scale_factor=4, mode="bicubic")
#     near4 = torch.nn.Upsample(scale_factor=4, mode="nearest")
#     eval_output = up4(eval_output.cpu())
#     eval_label = near4(eval_label.cpu().view(-1, 1, LABEL_SIZE, LABEL_SIZE).float()).squeeze().long()
#     print("New shapes {} {}".format(eval_output.shape, eval_label.shape))

#     metric = compute_mIOU(pred=eval_output, label=eval_label, n_cls=151, ignore_index=0)
#     print("Metric {}".format(metric))

#     eval_masks = eval_output.argmax(dim=1)
#     print("Masks shape {}".format(eval_masks.shape))

#     eval_masks = unpad(eval_masks, eval_size)
#     # eval_gt = unpad(eval_gt, eval_size)
#     print("Original shapes {}\n{}".format([x.shape for x in eval_masks], [x.shape for x in eval_label]))

#     colors = torch.LongTensor([[1, 255, 0, 255], [56, 0, 255, 0]])
#     color_masks = [apply_color(x, colors) for x in eval_masks]
#     color_labels = [apply_color(x, colors) for x in eval_label]

#     for i in range(len(eval_label)):
#         gt, m = color_labels[i], color_masks[i]
#         Path("./test/{}/".format(i)).mkdir(parents=True, exist_ok=True)
#         save_img("./test/{}/label.png".format(i), img=gt.to(torch.uint8))
#         save_img("./test/{}/mask.png".format(i), img=m.to(torch.uint8))

@timeit
def eval_list_gt():

    eval_data = ClassDataset(name=DATA_NAME, split="training", size=BATCH_SIZE*5)

    img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=IMG_SIZE, crop_size=IMG_SIZE)
    label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=LABEL_SIZE, crop_size=LABEL_SIZE)
    collate = FullClassCollator(img_transform=img_transform, label_transform=label_transform, return_tensors="pt", padding=True)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, collate_fn=collate)

    model = BaseModel(patch=PATCH_SIZE, in_size=IMG_SIZE, out_size=LABEL_SIZE, dropout=DROPOUT, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    eval_loss, eval_output, eval_label, eval_size, eval_gt = evaluate(model, eval_loader, criterion)
    print("Eval loss {}".format(eval_loss))
    print("Previous shapes {}\n{}".format(eval_output.shape, [x.shape for x in eval_gt]))

    eval_output, eval_gt = prepare_for_gt_metrics(eval_output.cpu(), eval_gt, eval_size)
    print("New shapes {} {}".format([x.shape for x in eval_output], [x.shape for x in eval_gt]))

    metric = compute_mIOU_list(pred=eval_output, label=eval_gt, n_cls=151, ignore_index=0)
    print("Metric {}".format(metric))

    eval_masks = [x.argmax(dim=0) for x in eval_output]
    print("Masks shape {}".format([x.shape for x in eval_masks]))

    colors = torch.LongTensor([[1, 255, 0, 255], [56, 0, 255, 0]])
    color_masks = [apply_color(x, colors) for x in eval_masks]
    color_labels = [apply_color(x, colors) for x in eval_gt]

    for i in range(len(eval_gt)):
        gt, m = color_labels[i], color_masks[i]
        Path("./test/{}/".format(i)).mkdir(parents=True, exist_ok=True)
        save_img("./test/{}/label.png".format(i), img=gt.to(torch.uint8))
        save_img("./test/{}/mask.png".format(i), img=m.to(torch.uint8))    

@timeit
def eval_list_label():

    eval_data = ClassDataset(name=DATA_NAME, split="training", size=BATCH_SIZE*5)

    img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", size=IMG_SIZE, crop_size=IMG_SIZE)
    label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False, size=LABEL_SIZE, crop_size=LABEL_SIZE)
    collate = FullClassCollator(img_transform=img_transform, label_transform=label_transform, return_tensors="pt", padding=True)
    eval_loader = DataLoader(dataset=eval_data, batch_size=BATCH_SIZE, collate_fn=collate)

    model = BaseModel(patch=PATCH_SIZE, in_size=IMG_SIZE, out_size=LABEL_SIZE, dropout=DROPOUT, device=DEVICE).to(DEVICE)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    eval_loss, eval_output, eval_label, eval_size, eval_gt = evaluate(model, eval_loader, criterion)
    print("Eval loss {}".format(eval_loss))
    print("Previous shapes {}\n{}".format(eval_output.shape, [x.shape for x in eval_gt]))

    # up4 = torch.nn.Upsample(scale_factor=4, mode="bicubic")
    # near4 = torch.nn.Upsample(scale_factor=4, mode="nearest")
    # eval_output = up4(eval_output.cpu())
    # eval_label = near4(eval_label.cpu().view(-1, 1, LABEL_SIZE, LABEL_SIZE).float()).squeeze().long()
    # print("New shapes {} {}".format(eval_output.shape, eval_label.shape))

    # eval_output = [x for x in eval_output]
    # eval_label = [x for x in eval_label]

    eval_output, eval_label = prepare_for_label_metrics(eval_output.cpu(), eval_label.cpu(), scale_factor=4)
    print("New shapes {} {}".format([x.shape for x in eval_output], [x.shape for x in eval_gt]))

    metric = compute_mIOU_list(pred=eval_output, label=eval_label, n_cls=151, ignore_index=0)
    print("Metric {}".format(metric))

    eval_masks = [x.argmax(dim=0) for x in eval_output]
    print("Masks shape {}".format([x.shape for x in eval_masks]))

    colors = torch.LongTensor([[1, 255, 0, 255], [56, 0, 255, 0]])
    color_masks = [apply_color(x, colors) for x in eval_masks]
    color_labels = [apply_color(x, colors) for x in eval_gt]

    for i in range(len(eval_gt)):
        gt, m = color_labels[i], color_masks[i]
        Path("./test/{}/".format(i)).mkdir(parents=True, exist_ok=True)
        save_img("./test/{}/label.png".format(i), img=gt.to(torch.uint8))
        save_img("./test/{}/mask.png".format(i), img=m.to(torch.uint8))    


def evaluate(model, eval_loader, criterion):
    eval_progress = tqdm(range(len(eval_loader)), desc="Evaluation", leave=False)

    all_eval_out = None
    all_eval_labels = None
    all_eval_size = None
    all_eval_loss = None
    all_eval_gt = None

    
    for data in eval_loader:
        inputs, mappings, originals = data
        labels, sizes, gts = inputs["label"], inputs["size"], originals["label"]
        inputs = {k:v.to(DEVICE) for k,v in inputs.items()}

        with torch.no_grad():
            _, _, outputs = model(inputs)
            eval_progress.update()
            loss = criterion(outputs, inputs["label"])

        all_eval_out = torch.concat([all_eval_out, outputs]) if all_eval_out is not None else outputs
        all_eval_labels = torch.concat([all_eval_labels, labels]) if all_eval_labels is not None else labels
        all_eval_size = torch.concat([all_eval_size, sizes]) if all_eval_size is not None else sizes
        all_eval_loss = torch.concat([all_eval_loss, loss.unsqueeze(0)]) if all_eval_loss is not None else loss.unsqueeze(0)
        all_eval_gt = all_eval_gt + gts if all_eval_gt is not None else gts


    return all_eval_loss.mean().item(), all_eval_out, all_eval_labels, all_eval_size, all_eval_gt

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

if __name__ == "__main__":

    # eval_tensor_gt()

    # eval_tensor_label()

    eval_list_gt()

    # eval_list_label()