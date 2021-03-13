import os
import cv2
import glob
import random
from PIL import Image
import xml.dom.minidom
import torch
from torch.utils.data import Dataset
from .utils import one_hot_embedding


class BirdDetection(Dataset):
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

        target = {"boxes": ann, "labels": lbl}

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


class BirdClassification(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        def get_image_list(root_dir):
            image_path_list = []
            for directory in os.listdir(root_dir):
                root = os.path.join(root_dir, directory)
                for file in glob.glob(os.path.join(root, "*.png")):
                    image_path_list.append(os.path.join(root, file))
            return image_path_list

        self.root_dir = root_dir

        self.foreground_dir = os.path.join(self.root_dir, "1")
        self.background_dir = os.path.join(self.root_dir, "0")

        self.image_path_list = get_image_list(self.foreground_dir)
        self.positive_instances = len(self.image_path_list)
        self.image_path_list += get_image_list(self.background_dir)
        self.negative_instances = len(self.image_path_list) - self.positive_instances
        random.shuffle(self.image_path_list)

        self.class_dic = {"bg": 0, "fg": 1}
        self.classes = 2

        self.image_size = None

    def __len__(self):
        return len(self.image_path_list)

    def get_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if self.image_size is None:
            self.image_size = image.size
        assert image.size == self.image_size

        return image

    def get_tags(self, image_path):
        if self.foreground_dir in image_path:
            return ["fg"]
        return ["bg"]

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]

        image = self.get_image(image_path)

        label = one_hot_embedding(
            [self.class_dic[tag] for tag in self.get_tags(image_path)], self.classes
        )

        if self.transform:
            image = self.transform(image)

        return image, {"img_cls_labels": label}

    def collate_fn(self, batch):
        imgs = [item[0] for item in batch]
        trgts = [item[1] for item in batch]

        return [imgs, trgts]
