import PIL

from data.dataset import SegmentationDataset
from data.collator import Collator
from data.utils import save_img

import torch

def test_SegmentationDataset():
    from torch.utils.data import DataLoader
    from transformers import CLIPFeatureExtractor, CLIPTokenizerFast

    data = SegmentationDataset(name="ade20k", split="training")

    img_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16")
    label_transform = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16", image_mean=[0, 0, 0], image_std=[1, 1, 1], resample=PIL.Image.Resampling.NEAREST, do_convert_rgb=False)
    txt_transform = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    collate = Collator(img_transform=img_transform, txt_transform=txt_transform, label_transform=label_transform, return_tensors="pt", padding=True)
    loader = DataLoader(dataset=data, batch_size=20, collate_fn=collate)

    index = 10
    colors = torch.LongTensor([[1, 255, 0, 0], [12, 0, 0, 255]])

    i = iter(loader)
    inputs, classes, originial = next(i)
    print({k: v.shape for k,v in inputs.items()})
    print({k: len(v) for k, v in classes.items()})
    print({k: len(v) for k, v in originial.items()})

    print(originial["label"][index].unique())


    save_img(file="224_img.png", img=inputs["pixel_values"][index])
    save_img(file="224_label.png", img=inputs["label"][index].type(torch.uint8))
    save_img(file="224_label_color.png", img=change_colors(inputs["label"][index], colors).type(torch.uint8))
    
    save_img(file="real_img.png", img=originial["img"][index])
    save_img(file="real_label.png", img=originial["label"][index].type(torch.uint8))
    save_img(file="real_label_color.png", img=change_colors(originial["label"][index], colors).type(torch.uint8))


def change_colors(x, colors):
    # x --> [H, W]
    # colors -> [[class_id, dim1, dim2, dim3], ...]
    mask = (x.flatten() == colors[:, :1])
    dim1 = (mask * colors[:, 1:2]).sum(0, keepdim=True).reshape(x.shape).unsqueeze(0)
    dim2 = (mask * colors[:, 2:3]).sum(0, keepdim=True).reshape(x.shape).unsqueeze(0)
    dim3 = (mask * colors[:, 3:]).sum(0, keepdim=True).reshape(x.shape).unsqueeze(0)
    dims = torch.concat([dim1, dim2, dim3])
    expand_mask = (1 - mask.sum(0)).reshape(x.shape).expand(3, -1, -1)
    return (expand_mask * x) + dims

if __name__ == "__main__":

    test_SegmentationDataset()