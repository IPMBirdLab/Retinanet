import os
import cv2
import glob
import random
import json
from PIL import Image
import xml.dom.minidom
import torch
from torch.utils.data import Dataset
from .utils import one_hot_embedding
from ..utils import create_directory


class BirdDetection(Dataset):
    def __init__(self, images_dir=None, annotations_dir=None, transform=None):
        if images_dir is not None or annotations_dir is not None:
            self.init_dataset(images_dir, annotations_dir)

        self.transforms = transform

    def init_dataset(self, images_dir="./data", annotations_dir="./ann"):
        self.files_name = os.listdir(images_dir)
        self.images_dir = images_dir
        self.annotaions_dir = annotations_dir

    def get_state_dict(self):
        return {
            "files_name": self.files_name,
            "images_dir": self.images_dir,
            "annotations_dir": self.annotaions_dir,
        }

    def set_state_dict(self, state_dict):
        self.files_name = state_dict["files_name"]
        self.images_dir = state_dict["images_dir"]
        self.annotaions_dir = state_dict["annotations_dir"]

    def load_state_dict(self, path):
        with open(path, "r", encoding="utf-8") as f:
            json_srialized = f.read()
            state_dict = json.loads(json_srialized)

            self.set_state_dict(state_dict)

            return state_dict

    def save_state_dict(self, path, state_dict=None):
        if state_dict is None:
            state_dict = self.get_state_dict()
        json_serialized = json.dumps(state_dict)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_serialized)

    def save(self, path, file_name=None):
        if file_name is None:
            file_name = "state_dict"
        create_directory(path)
        self.save_state_dict(os.path.join(path, f"{file_name}.json"))

    def load(self, path, file_name=None):
        if file_name is None:
            file_name = "state_dict"
        st_path = os.path.join(path, f"{file_name}.json")
        if not os.path.exists(st_path):
            raise FileNotFoundError
        self.load_state_dict(st_path)

    def subset(self, indices):
        st_dict = self.get_state_dict()
        st_dict["files_name"] = [st_dict["files_name"][idx] for idx in indices]
        new_subset = BirdDetection()
        new_subset.set_state_dict(st_dict)
        return new_subset

    def __len__(self):
        return len(self.files_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name, _ = os.path.splitext(self.files_name[idx])
        img_path = os.path.join(self.images_dir, file_name + ".png")
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

    @staticmethod
    def collate_fn(batch):
        imgs = [item[0] for item in batch]
        trgts = [item[1] for item in batch]

        return [imgs, trgts]


class BirdClassification(Dataset):
    def __init__(self, root_dir=None, transform=None):
        if root_dir is not None:
            self.init_dataset(root_dir)

        self.transform = transform

        self.class_dic = {"bg": 0, "fg": 1}
        self.classes = 2

        self.image_size = None

    def init_dataset(self, root_dir):
        self.root_dir = root_dir

        self.foreground_dir = os.path.join(self.root_dir, "1")
        self.background_dir = os.path.join(self.root_dir, "0")

        self.image_path_list = self._get_image_list(self.foreground_dir)
        positive_instances = len(self.image_path_list)
        self.image_path_list += self._get_image_list(self.background_dir)
        negative_instances = len(self.image_path_list) - positive_instances
        random.shuffle(self.image_path_list)

    def _get_image_list(self, root_dir):
        image_path_list = []
        for directory in os.listdir(root_dir):
            root = os.path.join(root_dir, directory)
            for file in glob.glob(os.path.join(root, "*.png")):
                image_path_list.append(os.path.join(root, file))
        return image_path_list

    def get_state_dict(self):
        return {
            "image_path_list": self.image_path_list,
            "root_dir": self.root_dir,
            "fdir": self.foreground_dir,
            "bdir": self.background_dir,
        }

    def set_state_dict(self, state_dict):
        self.image_path_list = state_dict["image_path_list"]
        self.root_dir = state_dict["root_dir"]
        self.foreground_dir = state_dict["fdir"]
        self.background_dir = state_dict["bdir"]

    def load_state_dict(self, path):
        with open(path, "r", encoding="utf-8") as f:
            json_srialized = f.read()
            state_dict = json.loads(json_srialized)

            self.set_state_dict(state_dict)

            return state_dict

    def save_state_dict(self, path, state_dict=None):
        if state_dict is None:
            state_dict = self.get_state_dict()
        json_serialized = json.dumps(state_dict)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_serialized)

    def save(self, path, file_name=None):
        if file_name is None:
            file_name = "state_dict"
        create_directory(path)
        self.save_state_dict(os.path.join(path, f"{file_name}.json"))

    def load(self, path, file_name=None):
        if file_name is None:
            file_name = "state_dict"
        st_path = os.path.join(path, f"{file_name}.json")
        if not os.path.exists(st_path):
            raise FileNotFoundError
        self.load_state_dict(st_path)

    def subset(self, indices):
        st_dict = self.get_state_dict()
        st_dict["image_path_list"] = [
            st_dict["image_path_list"][idx] for idx in indices
        ]
        new_subset = BirdClassification()
        new_subset.set_state_dict(st_dict)
        return new_subset

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

    @staticmethod
    def collate_fn(batch):
        imgs = [item[0] for item in batch]
        trgts = [item[1] for item in batch]

        return [imgs, trgts]
