import os
from typing import Callable

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from data.utils import create_text_file, read_txt_file

ADE20K_PATH = os.getcwd() + "/data/ade20k/data/ADEChallengeData2016/"

class SegmentationDataset(Dataset):

    def __init__(self, name: str, split: str = "validation") -> None:
        
        self.split = split

        self.img_folder = ADE20K_PATH + "images/" + self.split + "/"
        self.label_folder = ADE20K_PATH + "annotations/" + self.split + "/"
        
        self.class_info = open(ADE20K_PATH+"objectInfo150.txt", "r").readlines()
        # class n --> self.class_info[n-1]
        self.class_info = [c.strip("\n").split("\t")[-1].split(", ") for c in self.class_info][1:]

        if name == "ade20k":
            file_name = split + ".txt"
            if file_name in os.listdir(ADE20K_PATH):
                data_file = ADE20K_PATH + split + ".txt"
                self.data = read_txt_file(file=data_file)
            else:
                data_file = create_text_file(folder=ADE20K_PATH, image_path=self.img_folder, label_path=self.img_folder, split=self.split)
                self.data = read_txt_file(file=data_file)
        else:
            self.data = None

    def __getitem__(self, index) -> tuple:
        img_path, label_path = self.data[index]

        img = read_image(path=self.img_folder+img_path, mode=ImageReadMode.RGB)
        label = read_image(path=self.label_folder+label_path, mode=ImageReadMode.GRAY).squeeze()
        size = torch.LongTensor([label.size()])

        class_id = label.unique().tolist()
        # class_id.remove(0) #since it is not a class
        # first name if multiple names for a single class
        class_txt = [self.class_info[c-1][0] if c != 0 else "none" for c in class_id]

        return img, label, class_id, class_txt, size
    
    def __len__(self):
        return len(self.data)

    
    def text_id_mapping(self):
        
        x = open(ADE20K_PATH+"objectInfo150.txt", "r").readlines()
        info = [c.strip("\n").split("\t") for c in x]
        d = {info[i][-1].split(", ")[0]: dict(id=int(info[i][0]), texts=info[i][-1].split(", ")) for i in range(1, len(info))}
        return d

class ClassDataset(Dataset):

    def __init__(self, name: str, split: str = "validation", size: int = None) -> None:
        
        self.split = split

        self.img_folder = ADE20K_PATH + "images/" + self.split + "/"
        self.label_folder = ADE20K_PATH + "annotations/" + self.split + "/"
        
        self.info_dict = self.text_id_mapping()

        self.class_info = open(ADE20K_PATH+"objectInfo150.txt", "r").readlines()
        # class n --> self.class_info[n-1]
        self.class_info = [c.strip("\n").split("\t")[-1].split(", ") for c in self.class_info][1:]

        if name == "ade20k":
            file_name = split + ".txt"
            if file_name in os.listdir(ADE20K_PATH):
                data_file = ADE20K_PATH + split + ".txt"
                self.data = read_txt_file(file=data_file)
            else:
                data_file = create_text_file(folder=ADE20K_PATH, image_path=self.img_folder, label_path=self.img_folder, split=self.split)
                self.data = read_txt_file(file=data_file)
        else:
            self.data = None

        if size:
            self.data = self.data[0: size]

    def __getitem__(self, index) -> tuple:
        img_path, label_path = self.data[index]

        img = read_image(path=self.img_folder+img_path, mode=ImageReadMode.RGB)
        label = read_image(path=self.label_folder+label_path, mode=ImageReadMode.GRAY).squeeze()
        size = torch.LongTensor([label.size()])

        class_id = label.unique().tolist()
        class_id.remove(0) #since it is not a class
        # first name if multiple names for a single class
        txt = [self.class_info[c-1][0] if c != 0 else "none" for c in class_id]

        return img, label, size, txt
        return img, label, class_id, class_txt, size
    
    def __len__(self):
        return len(self.data)
    
    def text_id_mapping(self):
        x = open(ADE20K_PATH+"objectInfo150.txt", "r").readlines()
        info = [c.strip("\n").split("\t") for c in x]
        d = {info[i][-1].split(", ")[0]: dict(id=int(info[i][0]), texts=info[i][-1].split(", ")) for i in range(1, len(info))}
        return d

