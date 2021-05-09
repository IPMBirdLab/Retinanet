import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from retinanet.model.detection import retinanet_resnet50_fpn
from retinanet.model.utils import load_chpt
from retinanet.datasets.transforms import Compose, Normalize, ToTensor
from retinanet.datasets.bird import BirdDetection
from retinanet.datasets.utils import train_val_split, TransformDatasetWrapper

from retinanet.utils import create_directory

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse


#######################################################
# Device
#######################################################
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print("Torch Using device:", device)

###################################################################################
# Arguments
###################################################################################
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument("--data_dir", default="../../dataset", type=str)
parser.add_argument("--num_workers", default=1, type=int)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--use_p_of_data", default=0.5, type=float)

parser.add_argument("--pretrained", default="", type=str)
parser.add_argument("--pretrained_backend", action="store_true")

parser.add_argument("--log_dir", default="experiments", type=str)
parser.add_argument("--tag", default="", type=str)
args = parser.parse_args()
###################################################################################


def dump_results_dict(logs_dict, logs_path):
    res = pd.DataFrame.from_dict(logs_dict)
    res.to_csv(logs_path, float_format="%1.5f", index=False)


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = [float(np.mean(self.data_dic[key])) for key in keys]
        if clear:
            self.clear()

        if len(dataset) == 1:
            dataset = dataset[0]

        return dataset

    def clear(self):
        self.data_dic = {key: [] for key in self.keys}


def batched_nms(boxes, scores, labels=None, iou_threshold=0.5):
    """Implementation based on :
    https://github.com/pytorch/vision/issues/392#issuecomment-545809954

    Parameters
    ----------
    boxes : torch.Tensor[batch_size, N, 4]
    scores : torch.Tensor[batch_size, N]
    labels : torch.Tensor[batch_size, N]
    iou_threshold : float


    Returns
    -------
    [type]
        [description]
    """
    batch_size, N, _ = boxes.shape
    indices = torch.arange(batch_size, device=boxes.device)
    if labels is None:
        indices = indices[:, None].expand(batch_size, N).flatten()
    else:
        indices = (
            (labels + 1) * (indices[:, None].expand(batch_size, N) + 1)
        ).flatten()
    boxes_flat = boxes.flatten(0, 1)
    scores_flat = scores.flatten()
    indices_flat = torchvision.ops.boxes.batched_nms(
        boxes_flat, scores_flat, indices, iou_threshold
    )

    keep_indices = torch.stack([indices_flat // batch_size, indices_flat % batch_size])
    return keep_indices


def match_predictions(boxes, gt_boxes):
    match_matrix = torchvision.ops.box_iou(boxes, gt_boxes)
    matched_pred_scores, _ = match_matrix.max(dim=1)
    return matched_pred_scores


def _AP_metric(boxes, scores, labels, gt_boxes):
    """Inspiered by:
    https://github.com/pytorch/tnt/blob/master/torchnet/meter/apmeter.py
    """
    ap_list = []

    for (
        boxes_per_img,
        scores_per_img,
        labels_per_img,
        gt_boxes_per_img,
    ) in zip(boxes, scores, labels, gt_boxes):
        if boxes_per_img.size(0) == 0:
            ap_list.append(torch.zeros(1).to(boxes_per_img.device))
            continue
        keep = torchvision.ops.batched_nms(
            boxes_per_img, scores_per_img, labels_per_img, 0.5
        )
        boxes_per_img, scores_per_img = (
            boxes_per_img[keep],
            scores_per_img[keep],
        )

        match_scores = match_predictions(boxes_per_img, gt_boxes_per_img)
        _, match_indices = torch.sort(scores_per_img, dim=0, descending=True)
        match_scores = match_scores[match_indices]
        predicted_truth = torch.where(match_scores > 0.5, 1, 0).to(match_scores.device)

        # compute the true-positive sums
        tp = predicted_truth.float().cumsum(0)
        # create ranks range
        rg = torch.arange(1, tp.size(0) + 1).float().to(match_scores.device)
        # compute precision curve
        precision = tp.div(rg)
        # compute average precision
        ap = precision[match_scores.bool()].sum() / max(float(match_scores.sum()), 1)
        ap_list.append(ap)
    return ap_list


def _evaluate(model, loader):
    model.eval()

    val_meter = Average_Meter(["loss", "classification_loss", "bbox_regression_loss"])
    ap_meter = Average_Meter(["AP"])
    with torch.no_grad():
        for step, (images, targets) in enumerate(tqdm(loader)):
            ###############################################################################
            # Normal
            ###############################################################################
            losses, detections, _ = model(images, targets)

            loss = losses["classification"] + losses["bbox_regression"]

            # Batched_nms (useless)
            # boxes, scores, labels = (
            #     torch.stack([det["boxes"] for det in detections]),
            #     torch.stack([det["scores"] for det in detections]),
            #     torch.stack([det["labels"] for det in detections]),
            # )
            # keep = batched_nms(boxes, scores, labels, 0.5)
            # boxes, scores = boxes[keep], scores[keep]

            boxes, scores, labels = (
                [det["boxes"] for det in detections],
                [det["scores"] for det in detections],
                [det["labels"] for det in detections],
            )

            gt_boxes, gt_labels = (
                [lbl["boxes"] for lbl in targets],
                [lbl["labels"] for lbl in targets],
            )

            ap_list = _AP_metric(boxes, scores, labels, gt_boxes)
            for ap in ap_list:
                ap_meter.add({"AP": ap.item()})

            val_meter.add(
                {
                    "loss": loss.item(),
                    "classification_loss": losses["classification"].item(),
                    "bbox_regression_loss": losses["bbox_regression"].item(),
                }
            )

    model.train()

    return val_meter.get() + [ap_meter.get()]


def evaluate(model, val_loader):
    ################################################################################
    # Evaluate
    ################################################################################
    loss, cls_loss, bbox_loss, mAP = _evaluate(model, val_loader)

    logs_dict = {
        "loss": [loss],
        "cls_loss": [cls_loss],
        "bbox_loss": [bbox_loss],
        "mAP": [mAP],
    }

    dump_results_dict(
        logs_dict, os.path.join(args.log_dir, f"logs/logs_dict_{args.tag}.txt")
    )


if __name__ == "__main__":
    transform = Compose(
        [
            ToTensor(device),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = BirdDetection(
        images_dir=os.path.join(args.data_dir, "test"),
        annotations_dir=os.path.join(args.data_dir, "test"),
    )

    print(f"\nDataset size :   {len(dataset)}")

    dataset = TransformDatasetWrapper(dataset, transform)

    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=BirdDetection.collate_fn,
        drop_last=True,
        shuffle=True,
    )

    model = retinanet_resnet50_fpn(
        num_classes=2,
        pretrained=args.pretrained_backend,
        pretrained_backbone=args.pretrained_backend,
    ).to(device)

    if args.pretrained != "":
        print(f"Using pretrained model : {args.pretrained}")
        model = load_chpt(model, args.pretrained)

    evaluate(model, loader)
