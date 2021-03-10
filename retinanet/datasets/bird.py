import os
import cv2
import xml.dom.minidom
import torch
from torch.utils.data import Dataset


class BirdDataset(Dataset):
    def __init__(self, image_dir="./data", annotations_dir="./ann", transform=None):
        self.files_name = os.listdir(image_dir)
        self.image_dir = image_dir
        self.annotaions_dir = annotations_dir
        self.transforms = transform

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, _ = os.path.splitext(self.files_name[idx])
        img_path = os.path.join(self.image_dir, file_name + ".png")
        xml_path = os.path.join(self.annotaions_dir, file_name + ".xml")

        img = cv2.imread(img_path)
        ann = self.read_annotaions(xml_path)
        lbl = [1 for _ in range(len(ann))]

        target = {"boxes": torch.tensor(ann), "labels": torch.tensor(lbl)}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def read_annotaions(self, xml_path):
        res = []

        dom = xml.dom.minidom.parse(xml_path)
        # root = dom.documentElement
        objects = dom.getElementsByTagName("object")
        for obj in objects:
            bndbox = obj.getElementsByTagName("bndbox")[0]
            xmin = bndbox.getElementsByTagName("xmin")[0]
            ymin = bndbox.getElementsByTagName("ymin")[0]
            xmax = bndbox.getElementsByTagName("xmax")[0]
            ymax = bndbox.getElementsByTagName("ymax")[0]
            xmin_data = xmin.childNodes[0].data
            ymin_data = ymin.childNodes[0].data
            xmax_data = xmax.childNodes[0].data
            ymax_data = ymax.childNodes[0].data
            res.append([int(xmin_data), int(ymin_data), int(xmax_data), int(ymax_data)])

        return res

    def collate_fn(self, batch):
        imgs = [item[0] for item in batch]
        trgts = [item[1] for item in batch]

        return [imgs, trgts]
