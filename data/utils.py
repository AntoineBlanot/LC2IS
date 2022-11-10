import os

import torch
import torchvision.transforms.functional as F

def create_text_file(folder, image_path, label_path, split):
    imgs = os.listdir(image_path)
    labels = os.listdir(label_path)
    label_files = [l.split("/")[-1].split(".")[-2] for l in labels]

    # checking
    c = 0
    for img in imgs:
        if img.split("/")[-1].split(".")[-2] not in label_files:
            print(f"Image {img} has no corresponding annotation !!")
            c += 1
    print(f"Number of missing annotation: {c}")

    # Write text file (img_path label_path)
    file_name = folder + split + ".txt"
    with open(file_name, "w") as w:
        w.writelines([f"{l}.jpg {l}.png\n" for l in label_files])
    
    print(f"File has been created in {file_name}")
    return file_name

def read_txt_file(file: str):
    data = open(file, "r").readlines()
    data = [line.strip("\n").split(" ") for line in data]
    return data

def save_img(file: str, img: torch.Tensor, size: torch.Tensor = None):
    """Save a torch.Tensor to an image file. Resize it if size is given."""
    if size is not None:
        img = img[0: size[0], 0: size[1]]
    F.to_pil_image(img).save(file)