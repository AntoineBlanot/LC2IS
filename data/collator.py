from typing import Callable, List, Any, Dict, Tuple

import torch
import torch.nn.functional as F

class OldCollator():

    def __init__(self, pad_value: int = 0, img_transform: Callable = None, label_transform: Callable = None, txt_transform: Callable = None, **args) -> None:
        self.pad_value = pad_value
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.text_transform = txt_transform
        self.args = args

    def __call__(self, features: List[Any]) -> Dict[str, Any]:
        batch_img, batch_label, batch_class_id, batch_class_txt, batch_size = [list(f) for f in zip(*features)]
        
        # sizes as np.ndarray
        size = torch.cat(batch_size, dim=0)

        # image transformation
        if self.img_transform:
            img = self.img_transform(batch_img, **self.args)

        # label transformation
        if self.label_transform:
            new_batch_label = [label.expand(3, -1, -1) for label in batch_label]
            label = self.label_transform(new_batch_label, **self.args).pixel_values[:, :1, :, :].squeeze()
            label = (label * 255).long()
        else:
            # label padding (pad on the right and bottom)
            label = [
                F.pad(label, pad=(0, 1024 - s[1], 0, 1024 - s[0]), mode="constant", value=self.pad_value).unsqueeze(0)
                for label, s in zip(batch_label, size)
            ]
            label = torch.cat(label, dim=0)
            
        # id classes transformation
        batch_unique_classes = label.unique()
        # Tensor like [[old_class, new_class], ...]
        id_mapping = torch.concat([batch_unique_classes.unsqueeze(0).transpose(0,1), torch.arange(len(batch_unique_classes), dtype=torch.long).unsqueeze(0).transpose(0,1)], dim=1)
        label = self.change_ids(label, id_mapping=id_mapping)

        # textual classes transformation
        # dict like {text: old_class} for all classes
        text_id_mapping = {text: class_id for text, class_id in zip(sum(batch_class_txt, []),sum(batch_class_id, []))}
        # dict like {text: old_class} for available classes only
        avaliable_txt_id = {text: i for text, i in text_id_mapping.items() if i in batch_unique_classes}
        # dict like {text: new_class} for available classes only
        new_txt_id = {text: id_mapping[id_mapping[:, 0] == i, :][0][1].item() for text, i in avaliable_txt_id.items()}
        # list of the availbale text in the batch
        new_txt = list(new_txt_id.keys())
        # test avaliable_txt_id
        if self.text_transform:
            txt = self.text_transform(new_txt, **self.args)
        return dict(**img, label=label, **txt, size=size), dict(old_new_id_mapping=id_mapping, text_new_id_mapping=new_txt_id), dict(img=batch_img, label=batch_label, text=batch_class_txt, class_id=batch_class_id)

    def change_ids(self, x, id_mapping):
        # print(x.unique(), len(x.unique()), x.shape)
        flattened_x = x.flatten()
        mask = (flattened_x == id_mapping[:, :1])
        flattened_x = (1 - mask.sum(dim=0)) * flattened_x + (mask * id_mapping[:,1:]).sum(dim=0)
        x = flattened_x.reshape(x.shape)
        # print(x.unique(), len(x.unique()), x.shape)
        return x


class JoinTextCollator():

    def __init__(self, img_transform: Callable, label_transform: Callable = None, txt_transform: Callable = None, join_text: str = ", ", pad_value: int = 0, **args) -> None:
        self.pad_value = pad_value
        self.join_text = join_text
        self.img_transform = img_transform
        self.label_transform = label_transform
        if self.label_transform is None:
            self.LABEL_SIZE = 1024
        self.text_transform = txt_transform
        self.args = args

    def __call__(self, features: List[Any]) -> Tuple[Dict]:
        img_list, label_list, size_list, class_texts_list, class_ids_list = [list(f) for f in zip(*features)]

        # image transformation
        if self.img_transform:
            imgs = self.img_transform(img_list, **self.args)

        # label transformation
        if self.label_transform:
            label_list_expand = [label.expand(3, -1, -1) for label in label_list]          # need to expand on RGB channel to use HuggingFace's Feature Extractor
            labels = self.label_transform(label_list_expand, **self.args)
            labels = (labels.pixel_values[:, 0, :, :] * 255).long()      # only take the first element on the expanded channel, also convert to int classes
        else:
            # if no transformation we have to pad labels to create Tensor
            label_list = [
                F.pad(label, pad=(0, self.LABEL_SIZE - s[1], 0, self.LABEL_SIZE - s[0]), mode="constant", value=self.pad_value).unsqueeze(0)
                for label, s in zip(label_list, size_list)
            ]
            labels = torch.cat(label_list, dim=0)

        # text transformation
        # for each image, join all the class_txt with a comma
        class_texts_join = [self.join_text.join(class_texts) for class_texts in class_texts_list] 
        if self.text_transform:
            texts = self.text_transform(class_texts_join, **self.args)

        # sizes
        sizes = torch.cat(size_list, dim=0)

        # format output
        inputs = dict(**imgs, label=labels, **texts)
        classes = dict(text=class_texts_list, id=class_ids_list)
        sizes = dict(size=sizes)
        originals = dict(img=img_list, label=label_list, size=size_list, text=class_texts_list, id=class_ids_list)
        
        return inputs, classes, sizes, originals


class FullClassesCollator():

    def __init__(self, img_transform: Callable, label_transform: Callable = None, txt_transform: Callable = None, join_text: str = ", ", pad_value: int = 0, **args) -> None:
        self.pad_value = pad_value
        self.join_text = join_text
        self.img_transform = img_transform
        self.label_transform = label_transform
        if self.label_transform is None:
            self.LABEL_SIZE = 1024
        self.text_transform = txt_transform
        self.args = args

    def __call__(self, features: List[Any]) -> Tuple[Dict]:
        img_list, label_list, size_list, class_texts_list, class_ids_list, classes_list = [list(f) for f in zip(*features)]

        # image transformation
        if self.img_transform:
            imgs = self.img_transform(img_list, **self.args)

        # label transformation
        if self.label_transform:
            label_list_expand = [label.expand(3, -1, -1) for label in label_list]          # need to expand on RGB channel to use HuggingFace's Feature Extractor
            labels = self.label_transform(label_list_expand, **self.args)
            labels = (labels.pixel_values[:, 0, :, :] * 255).long()      # only take the first element on the expanded channel, also convert to int classes
        else:
            # if no transformation we have to pad labels to create Tensor
            label_list = [
                F.pad(label, pad=(0, self.LABEL_SIZE - s[1], 0, self.LABEL_SIZE - s[0]), mode="constant", value=self.pad_value).unsqueeze(0)
                for label, s in zip(label_list, size_list)
            ]
            labels = torch.cat(label_list, dim=0)

        # text transformation
        classes = list(set(sum(classes_list, [])))
        if self.text_transform:
            texts = self.text_transform(classes, **self.args)

        # sizes
        sizes = torch.cat(size_list, dim=0)

        # format output
        inputs = dict(**imgs, label=labels, **texts)
        classes = dict(text=class_texts_list, id=class_ids_list)
        sizes = dict(size=sizes)
        originals = dict(img=img_list, label=label_list, size=size_list, text=class_texts_list, id=class_ids_list)
        
        return inputs, classes, sizes, originals


class ADE20KCollator():

    def __init__(self) -> None:
        pass

    def __call__(self, features: List[Any]) -> tuple:
        img_list, label_list, metas_list = [list(f) for f in zip(*features)]

        img = torch.cat(img_list, dim=0)
        label = torch.cat(label_list, dim=0)
        metas = metas_list

        inputs = dict(pixel_values=img, label=label)
        return inputs, metas
