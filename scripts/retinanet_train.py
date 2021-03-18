import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from retinanet.model.detection import retinanet_resnet50_fpn
from retinanet.datasets.transforms import Compose, Normalize, ToTensor
from retinanet.datasets.bird import BirdDetection
from retinanet.datasets.utils import train_val_split, TransformDatasetWrapper

import os
import numpy as np

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
parser.add_argument("--val_batch_size", default=None, type=int)
parser.add_argument("--max_epoch", default=1, type=int)
parser.add_argument("--train_percent", default=0.9, type=float)
parser.add_argument("--use_p_of_data", default=0.5, type=float)

parser.add_argument("--lr", default=0.01, type=float)

parser.add_argument("--pretrained", default="", type=str)
parser.add_argument("--pretrained_backend", action="store_true")

parser.add_argument("--log_dir", default="experiments", type=str)
parser.add_argument("--tag", default="", type=str)
args = parser.parse_args()
if args.val_batch_size is None:
    args.val_batch_size = args.batch_size
###################################################################################


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


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


def evaluate(model, loader):
    model.eval()

    val_meter = Average_Meter(["loss", "classification_loss", "bbox_regression_loss"])
    ap_meter = Average_Meter(["AP"])
    with torch.no_grad():
        for step, (images, targets) in enumerate(loader):
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

            for (
                boxes_per_img,
                scores_per_img,
                labels_per_img,
                gt_boxes_per_img,
                gt_labels_per_img,
            ) in zip(boxes, scores, labels, gt_boxes, gt_labels):
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
                predicted_truth = torch.where(match_scores > 0.5, 1, 0).to(
                    match_scores.device
                )

                # compute the true-positive sums
                tp = predicted_truth.float().cumsum(0)
                # create ranks range
                rg = torch.arange(1, tp.size(0) + 1).float().to(match_scores.device)
                # compute precision curve
                precision = tp.div(rg)
                # compute average precision
                ap = precision[match_scores.bool()].sum() / max(
                    float(match_scores.sum()), 1
                )
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


def _train(model, train_loader, val_loader):
    epochs = args.max_epoch

    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=args.lr * 1e-5
    )

    writer = SummaryWriter(
        os.path.join(args.log_dir, "logs/tensorboard"), filename_suffix=args.tag
    )
    train_meter = Average_Meter(["loss", "classification_loss", "bbox_regression_loss"])

    best_metric = 1000
    for epoch in range(1, epochs):
        for i_batch, batch in enumerate(train_loader):

            optimizer.zero_grad()

            losses, _ = model(*batch)

            loss = losses["classification"] + losses["bbox_regression"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            ################################################################################
            # Log
            ################################################################################
            train_meter.add(
                {
                    "loss": loss.item(),
                    "classification_loss": losses["classification"].item(),
                    "bbox_regression_loss": losses["bbox_regression"].item(),
                }
            )
            iteration = epoch * len(train_loader) + i_batch
            writer.add_scalar("Train/Losses/loss", loss.item(), iteration)
            writer.add_scalar(
                "Train/Losses/classification_loss",
                losses["classification"].item(),
                iteration,
            )
            writer.add_scalar(
                "Train/Losses/bbox_regression_loss",
                losses["bbox_regression"].item(),
                iteration,
            )

            print(
                "Epoch: {} | batch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}".format(
                    epoch,
                    i_batch + 1,
                    float(losses["classification"].item()),
                    float(losses["bbox_regression"].item()),
                    float(loss.item()),
                )
            )
        scheduler.step()
        writer.add_scalar("HP/lr", optimizer.param_groups[0]["lr"], epoch)
        ################################################################################
        # Evaluate
        ################################################################################
        loss, cls_loss, bbox_loss, mAP = evaluate(model, val_loader)
        tloss, tcls_loss, tbbox_loss = train_meter.get(clear=True)

        writer.add_scalars("Evaluate/Losses/loss", {"train": tloss, "val": loss}, epoch)
        writer.add_scalars(
            "Evaluate/Losses/classification_loss",
            {"train": tcls_loss, "val": cls_loss},
            epoch,
        )
        writer.add_scalars(
            "Evaluate/Losses/bbox_regression_loss",
            {"train": tbbox_loss, "val": bbox_loss},
            epoch,
        )
        writer.add_scalars(
            "Evaluate/Metrics/mAP",
            {"train": 0, "val": mAP},
            epoch,
        )
        print(
            "Evaluation -> Epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | mAP: {:.2f}".format(
                epoch, cls_loss, bbox_loss, loss, mAP
            )
        )

        if loss < best_metric:
            best_metric = loss
            base_dir = os.path.join(args.log_dir, "checkpoints")
            create_directory(base_dir)
            torch.save(
                model.state_dict(),
                os.path.join(base_dir, f"best_chpt_{args.tag}.pth"),
            )

    return model


if __name__ == "__main__":
    train_transform = Compose(
        [
            ToTensor(device),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = BirdDetection(
        image_dir=os.path.join(args.data_dir, "data"),
        annotations_dir=os.path.join(args.data_dir, "ann"),
    )

    train_idx, valid_idx = train_val_split(
        dataset, p=args.train_percent, use_p_of_data=args.use_p_of_data
    )

    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataset = Subset(TransformDatasetWrapper(dataset, train_transform), train_idx)
    val_dataset = Subset(TransformDatasetWrapper(dataset, train_transform), valid_idx)

    print(f"\nDataset size :   {len(dataset)}")
    print(f"Training subset:   {len(train_dataset)}")
    print(f"Validation subset: {len(val_dataset)}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    model = retinanet_resnet50_fpn(
        num_classes=2,
        pretrained=args.pretrained_backend,
        pretrained_backbone=args.pretrained_backend,
    )

    if args.pretrained != "":
        print(f"Using pretrained model : {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained))

    model = _train(model, train_loader, val_loader)

    base_dir = os.path.join(args.log_dir, "checkpoints")
    create_directory(base_dir)
    torch.save(
        model.state_dict(),
        os.path.join(base_dir, f"chpt_{args.tag}.pth"),
    )
