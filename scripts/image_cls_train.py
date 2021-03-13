import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from retinanet.model.detection import retinanet_resnet50_fpn
from retinanet.datasets.transforms import Compose, Normalize, ToTensor
from retinanet.datasets.bird import BirdClassification
from retinanet.datasets.utils import train_val_split, TransformDatasetWrapper

from retinanet.utils.gpu_profile import gpu_profile

import os
import sys
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


def calculate_metrics(preds, labels):
    true_preds = (preds == labels).float().sum(0)
    total = len(labels)
    total_predicted_positive = (preds == 1).float().sum(0)
    total_actual_positive = (labels == 1).float().sum(0)

    true_positive = ((preds == labels) * labels).float().sum(0)
    # false_positive = ((preds - labels) > 0).float().sum()

    acc = true_preds / total
    percision = true_positive / total_predicted_positive
    recall = true_positive / total_actual_positive
    f1 = (2 * percision * recall) / (percision + recall)

    return acc[1], percision[1], recall[1], f1[1]


def logits_to_preds(logits):
    return (logits > 0.5).float()


def outputs_to_logits(outputs):
    return nn.Sigmoid()(outputs)


def evaluate(model, loader):
    model.eval()

    val_meter = Average_Meter(
        ["loss", "image_classification_loss", "acc", "precision", "recall", "F1"]
    )
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            ###############################################################################
            # Normal
            ###############################################################################
            losses, _, cls_outputs = model(images, labels)

            loss = losses["img_classification"]

            predicted = logits_to_preds(outputs_to_logits(cls_outputs))

            labels = torch.cat(
                list(map(lambda x: x["img_cls_labels"].unsqueeze(0), labels)), 0
            )

            acc, precision, recall, f1 = calculate_metrics(predicted.detach(), labels)

            val_meter.add(
                {
                    "loss": loss.item(),
                    "image_classification_loss": losses["img_classification"].item(),
                    "acc": acc.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                    "F1": f1.item(),
                }
            )

    model.train()

    return val_meter.get()


def _train(model, train_loader, val_loader):
    epochs = args.max_epoch

    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
    # optimizer = optim.SGD(
    #     model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01, nesterov=False
    # )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=args.lr * 1e-5
    )

    writer = SummaryWriter(os.path.join(args.log_dir, "logs/tensorboard"))
    train_meter = Average_Meter(
        ["loss", "image_classification_loss", "acc", "precision", "recall", "F1"]
    )

    best_metric = 1000
    for epoch in range(1, epochs):
        for i_batch, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            losses, cls_outputs = model(images, labels)

            # loss = losses["classification"] + losses["bbox_regression"]
            loss = losses["img_classification"]
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            ################################################################################
            # Log
            ################################################################################
            predicted = logits_to_preds(outputs_to_logits(cls_outputs))
            labels = torch.cat(
                list(map(lambda x: x["img_cls_labels"].unsqueeze(0), labels)), 0
            )
            acc, precision, recall, f1 = calculate_metrics(predicted.detach(), labels)

            train_meter.add(
                {
                    "loss": loss.item(),
                    "image_classification_loss": losses["img_classification"].item(),
                    "acc": acc.item(),
                    "precision": precision.item(),
                    "recall": recall.item(),
                    "F1": f1.item(),
                }
            )
            iteration = epoch * len(train_loader) + i_batch
            writer.add_scalar("Train/Losses/loss", loss.item(), iteration)
            writer.add_scalar(
                "Train/Losses/image_classification_loss",
                losses["img_classification"].item(),
                iteration,
            )

            print(
                "Epoch: {} | batch: {}/{:.2f}% | Image Classification loss: {:1.5f} | Running loss: {:1.5f}".format(
                    epoch,
                    i_batch + 1,
                    ((i_batch + 1) / len(train_loader)) * 100,
                    float(losses["img_classification"].item()),
                    float(loss.item()),
                )
            )

        scheduler.step()
        writer.add_scalar("HP/lr", optimizer.param_groups[0]["lr"], epoch)
        ################################################################################
        # Evaluate
        ################################################################################
        loss, cls_loss, acc, precision, recall, f1 = evaluate(model, val_loader)
        tloss, tcls_loss, tacc, tprecision, trecall, tf1 = train_meter.get(clear=True)

        writer.add_scalars("Evaluate/Losses/loss", {"train": tloss, "val": loss}, epoch)
        writer.add_scalars(
            "Evaluate/Losses/image_classification_loss",
            {"train": tcls_loss, "val": cls_loss},
            epoch,
        )
        writer.add_scalars(
            "Evaluate/Metrics/Accuracy",
            {"train": tacc, "val": acc},
            epoch,
        )
        writer.add_scalars(
            "Evaluate/Metrics/Precision",
            {"train": tprecision, "val": precision},
            epoch,
        )
        writer.add_scalars(
            "Evaluate/Metrics/Recall",
            {"train": trecall, "val": recall},
            epoch,
        )
        writer.add_scalars(
            "Evaluate/Metrics/F1",
            {"train": tf1, "val": f1},
            epoch,
        )
        print(
            "Evaluation -> Epoch: {} | Image Classification loss: {:1.5f} | Running loss: {:1.5f} | Accuracy: {:1.5f} | F1: {:1.5f}".format(
                epoch,
                cls_loss,
                loss,
                acc,
                f1,
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
    # for GPU memory trace
    # sys.settrace(gpu_profile)

    train_transform = Compose(
        [
            ToTensor(device),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    dataset = BirdClassification(root_dir=args.data_dir)

    train_idx, valid_idx = train_val_split(
        dataset, p=args.train_percent, use_p_of_data=args.use_p_of_data
    )

    # train_sampler = SubsetRandomSampler(train_idx)
    # valid_sampler = SubsetRandomSampler(valid_idx)

    train_dataset = Subset(TransformDatasetWrapper(dataset, train_transform), train_idx)
    val_dataset = Subset(TransformDatasetWrapper(dataset, train_transform), valid_idx)

    print(f"\nDataset size :     {len(dataset)}")
    print(f"Training subset:     {len(train_dataset)}")
    print(f"Validation subset:   {len(val_dataset)}")
    print(f"\nBatches per epoch: {len(train_dataset)//args.batch_size}")

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
        drop_last=True,
        shuffle=False,
    )

    model = retinanet_resnet50_fpn(
        num_classes=2, pretrained=False, pretrained_backbone=False
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
