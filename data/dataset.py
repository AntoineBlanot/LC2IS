import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from data.utils import create_text_file, read_txt_file

ADE20K_DATA_DIR = os.getcwd() + "/data/ade20k/data/ADEChallengeData2016/"
ADE20K_DATA_INFO_FILE = ADE20K_DATA_DIR + "objectInfo150.txt"


class ADE20K_Dataset(Dataset):

    def __init__(self, split: str = "validation", size: int = None) -> None:
        
        self.split = split
        self.img_folder = ADE20K_DATA_DIR + "images/" + self.split + "/"
        self.label_folder = ADE20K_DATA_DIR + "annotations/" + self.split + "/" 
        self.mapping = self.get_mapping(info_file=ADE20K_DATA_INFO_FILE)

        data_file = split + ".txt"
        if data_file in os.listdir(ADE20K_DATA_DIR):
            print("Reusing already existing data file at path {}".format(ADE20K_DATA_DIR+data_file))
            data_file = ADE20K_DATA_DIR + data_file
            self.data = read_txt_file(file=data_file)
        else:
            print("Could not find existing data file in folder {}, creating a new one...".format(ADE20K_DATA_DIR))
            data_file = create_text_file(folder=ADE20K_DATA_DIR, image_path=self.img_folder, label_path=self.img_folder, split=self.split)
            self.data = read_txt_file(file=data_file)

        if size:
            self.data = self.data[0: size]

    def __getitem__(self, index) -> tuple:
        img_path, label_path = self.data[index]

        img = read_image(path=self.img_folder+img_path, mode=ImageReadMode.RGB)
        label = read_image(path=self.label_folder+label_path, mode=ImageReadMode.GRAY).squeeze()
        size = torch.LongTensor([label.size()])

        class_ids = label.unique().tolist()
        class_ids.remove(0)      # since it is not a class
        class_texts = [self.mapping[id - 1]["cls"] for id in class_ids]

        return img, label, size, class_texts, class_ids
    
    def __len__(self):
        return len(self.data)

    def get_mapping(self, info_file: str) -> List[Dict]:
        """Return a list of the mapping between class ID and their corresponding textual values"""
        x = open(info_file, "r").readlines()
        info = [c.strip("\n").split("\t") for c in x]
        mapping = [dict(id=int(info[i][0]), cls=info[i][-1].split(", ")[0], text_list=info[i][-1].split(", ")) for i in range(1, len(info))]
        return mapping
