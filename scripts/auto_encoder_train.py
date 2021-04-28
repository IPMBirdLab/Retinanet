import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

from retinanet.model.detection import retinanet_resnet50_fpn
from retinanet.model.utils import outputs_to_logits, logits_to_preds
from retinanet.datasets.transforms import Compose, Normalize, ToTensor, RandAugment
from retinanet.datasets.bird import BirdClassificationRegeneration
from retinanet.datasets.utils import train_val_split, TransformDatasetWrapper

from retinanet.utils import create_directory
from retinanet.utils.gpu_profile import gpu_profile

import os
import sys
import numpy as np
import pandas as pd

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
parser.add_argument("--load_from_json", action="store_true")
parser.add_argument("--num_workers", default=1, type=int)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument("--batch_size", default=2, type=int)
parser.add_argument("--accumulation_steps", default=1, type=int)
parser.add_argument("--val_batch_size", default=None, type=int)
parser.add_argument("--max_epoch", default=1, type=int)
parser.add_argument("--train_percent", default=0.9, type=float)
parser.add_argument("--use_p_of_data", default=0.5, type=float)

parser.add_argument("--opt", default="sgd", type=str)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--lr_delta", default=1e-5, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--lr_warmup", default=1e-1, type=float)
# SGD
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--nesterov", action="store_true")

parser.add_argument("--pretrained", default="", type=str)
parser.add_argument("--pretrained_backend", action="store_true")

parser.add_argument("--log_dir", default="experiments", type=str)
parser.add_argument("--tag", default="", type=str)
args = parser.parse_args()
if args.val_batch_size is None:
    args.val_batch_size = args.batch_size
###################################################################################


def dump_results_dict(logs_dict, logs_path):
    res_dict = {}
    for key in logs_dict.keys():
        res_dict[f"{key}_train"] = [logs_dict[key][0]]
        res_dict[f"{key}_val"] = [logs_dict[key][1]]

    res = pd.DataFrame.from_dict(res_dict)
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


def calculate_metrics(preds, labels):
    true_preds = (preds == labels).float().sum(0)
    total = len(labels)
    total_predicted_positive = (preds == 1).float().sum(0)
    total_actual_positive = (labels == 1).float().sum(0)

    true_positive = ((preds == labels) * labels).float().sum(0)
    # false_positive = ((preds - labels) > 0).float().sum()

    acc = true_preds / total
    percision = true_positive / torch.maximum(
        total_predicted_positive,
        torch.tensor([1, 1]).to(total_predicted_positive.device),
    )
    recall = true_positive / torch.maximum(
        total_actual_positive, torch.tensor([1, 1]).to(total_actual_positive.device)
    )
    f1 = (2 * percision * recall) / torch.maximum(
        (percision + recall), torch.tensor([1, 1]).to(percision.device)
    )

    return acc[1], percision[1], recall[1], f1[1]


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Note: Implementation is origially from :
            https://github.com/ildoonet/pytorch-gradual-warmup-lr

    Parameters
    ----------
    optimizer : torch.optim
        Wrapped optimizer.
    warmup_steps : int
        warmup duration. target learning rate is reached at warmup_steps, gradually.
    after_scheduler : _LRScheduler
        after warmup_steps, use this scheduler(eg. ReduceLROnPlateau)
    multiplier : float, optional
        target learning rate = base lr * multiplier if multiplier > 1.0.
        if multiplier = 1.0, lr starts from 0 and ends up with the base_lr, by default 1.0

    Raises
    ------
    ValueError
        multiplier should be greater thant or equal to 1.
    """

    def __init__(self, optimizer, warmup_steps, after_scheduler, multiplier=1.0):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.warmup_steps:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.warmup_steps)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.warmup_steps + 1.0)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.warmup_steps)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def evaluate(model, loader):
    model.eval()

    val_meter = Average_Meter(
        [
            "loss",
            "image_classification_loss",
            "regeneration_loss",
            "acc",
            "precision",
            "recall",
            "F1",
        ]
    )
    with torch.no_grad():
        for step, (images, labels) in enumerate(loader):
            ###############################################################################
            # Normal
            ###############################################################################
            losses, _, cls_outputs = model(images, labels)

            loss = losses["img_classification"] + losses["auto_encoder"]

            predicted = logits_to_preds(outputs_to_logits(cls_outputs))

            labels = torch.cat(
                list(map(lambda x: x["img_cls_labels"].unsqueeze(0), labels)), 0
            )

            acc, precision, recall, f1 = calculate_metrics(predicted.detach(), labels)

            val_meter.add(
                {
                    "loss": loss.item(),
                    "image_classification_loss": losses["img_classification"].item(),
                    "regeneration_loss": losses["auto_encoder"].item(),
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

    if args.opt == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.opt == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    total_steps = (epochs * len(train_loader)) // args.accumulation_steps

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * args.lr_delta
    )
    if args.lr_warmup != -1:
        scheduler = GradualWarmupScheduler(
            optimizer, total_steps * args.lr_warmup, scheduler
        )

    train_writer = SummaryWriter(
        os.path.join(args.log_dir, f"logs/tensorboard/{args.tag}/train"),
        filename_suffix=args.tag,
    )
    val_writer = SummaryWriter(
        os.path.join(args.log_dir, f"logs/tensorboard/{args.tag}/val"),
        filename_suffix=args.tag,
    )
    train_meter = Average_Meter(
        [
            "loss",
            "image_classification_loss",
            "regeneration_loss",
            "acc",
            "precision",
            "recall",
            "F1",
        ]
    )

    logs_dict = {
        "best_loss": [1000, 1000],
        "best_acc": [-1, -1],
        "best_f1": [-1, -1],
    }

    loss_value_dict = {"loss": 0, "img_classification": 0, "regeneration": 0}

    for epoch in range(0, epochs):
        for i_batch, (images, labels) in enumerate(train_loader):
            iteration = epoch * len(train_loader) + i_batch

            losses, cls_outputs = model(images, labels)

            loss = losses["img_classification"] + losses["auto_encoder"]
            # Normalize our loss (if averaged)
            loss = loss / args.accumulation_steps

            loss.backward()

            loss_value_dict["loss"] += loss.item()
            loss_value_dict["img_classification"] += (
                losses["img_classification"].item() / args.accumulation_steps
            )
            loss_value_dict["regeneration"] += (
                losses["auto_encoder"].item() / args.accumulation_steps
            )

            if (
                iteration + 1
            ) % args.accumulation_steps == 0 or args.accumulation_steps == 1:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            # log model metrics after each backprop step
            if (
                iteration + 1
            ) % args.accumulation_steps == 1 or args.accumulation_steps == 1:
                ################################################################################
                # Log
                ################################################################################
                predicted = logits_to_preds(outputs_to_logits(cls_outputs))
                labels = torch.cat(
                    list(map(lambda x: x["img_cls_labels"].unsqueeze(0), labels)), 0
                )
                acc, precision, recall, f1 = calculate_metrics(
                    predicted.detach(), labels
                )

                train_meter.add(
                    {
                        "loss": loss_value_dict["loss"],
                        "image_classification_loss": loss_value_dict[
                            "img_classification"
                        ],
                        "regeneration_loss": loss_value_dict["regeneration"],
                        "acc": acc.item(),
                        "precision": precision.item(),
                        "recall": recall.item(),
                        "F1": f1.item(),
                    }
                )
                train_writer.add_scalar(
                    "Train/Losses/loss",
                    loss_value_dict["loss"],
                    iteration // args.accumulation_steps,
                )
                train_writer.add_scalar(
                    "Train/Losses/image_classification_loss",
                    loss_value_dict["img_classification"],
                    iteration // args.accumulation_steps,
                )
                train_writer.add_scalar(
                    "Train/Losses/regeneration_loss",
                    loss_value_dict["regeneration"],
                    iteration // args.accumulation_steps,
                )

                train_writer.add_scalar(
                    "HP/lr",
                    optimizer.param_groups[0]["lr"],
                    iteration // args.accumulation_steps,
                )

                print(
                    "Epoch: {} | batch: {}/{:.2f}% | Image Classification loss: {:1.5f} | Regeneration loss: {:1.5f}".format(
                        epoch,
                        i_batch + 1,
                        ((i_batch + 1) / len(train_loader)) * 100,
                        float(losses["img_classification"].item()),
                        float(losses["auto_encoder"].item()),
                    )
                )
                loss_value_dict["loss"] = 0
                loss_value_dict["img_classification"] = 0
                loss_value_dict["regeneration"] = 0

        ################################################################################
        # Evaluate
        ################################################################################
        loss, cls_loss, regen_loss, acc, precision, recall, f1 = evaluate(
            model, val_loader
        )
        tloss, tcls_loss, tregen_loss, tacc, tprecision, trecall, tf1 = train_meter.get(
            clear=True
        )

        train_writer.add_scalar("Evaluate/Losses/loss", tloss, epoch)
        val_writer.add_scalar("Evaluate/Losses/loss", loss, epoch)
        train_writer.add_scalar(
            "Evaluate/Losses/image_classification_loss",
            tcls_loss,
            epoch,
        )
        val_writer.add_scalar(
            "Evaluate/Losses/image_classification_loss",
            cls_loss,
            epoch,
        )
        train_writer.add_scalar(
            "Evaluate/Losses/regeneration_loss",
            tregen_loss,
            epoch,
        )
        val_writer.add_scalar(
            "Evaluate/Losses/regeneration_loss",
            regen_loss,
            epoch,
        )
        train_writer.add_scalar(
            "Evaluate/Metrics/Accuracy",
            tacc,
            epoch,
        )
        val_writer.add_scalar(
            "Evaluate/Metrics/Accuracy",
            acc,
            epoch,
        )
        train_writer.add_scalar(
            "Evaluate/Metrics/Precision",
            tprecision,
            epoch,
        )
        val_writer.add_scalar(
            "Evaluate/Metrics/Precision",
            precision,
            epoch,
        )
        train_writer.add_scalar(
            "Evaluate/Metrics/Recall",
            trecall,
            epoch,
        )
        val_writer.add_scalar(
            "Evaluate/Metrics/Recall",
            recall,
            epoch,
        )
        train_writer.add_scalar(
            "Evaluate/Metrics/F1",
            tf1,
            epoch,
        )
        val_writer.add_scalar(
            "Evaluate/Metrics/F1",
            f1,
            epoch,
        )
        print(
            "Evaluation -> Epoch: {} | Image Classification loss: {:1.5f} | Regenerarion loss: {:1.5f} | Accuracy: {:1.5f} | F1: {:1.5f}".format(
                epoch,
                cls_loss,
                regen_loss,
                acc,
                f1,
            )
        )

        if tloss < logs_dict["best_loss"][0]:
            logs_dict["best_loss"][0] = tloss
        if tacc > logs_dict["best_acc"][0]:
            logs_dict["best_acc"][0] = tacc
        if tf1 > logs_dict["best_f1"][0]:
            logs_dict["best_f1"][0] = tf1
        if loss < logs_dict["best_loss"][1]:
            logs_dict["best_loss"][1] = loss
            base_dir = os.path.join(args.log_dir, "checkpoints")
            create_directory(base_dir)
            torch.save(
                model.state_dict(),
                os.path.join(base_dir, f"best_chpt_{args.tag}.pth"),
            )
        if acc > logs_dict["best_acc"][1]:
            logs_dict["best_acc"][1] = acc
        if f1 > logs_dict["best_f1"][1]:
            logs_dict["best_f1"][1] = f1

    dump_results_dict(
        logs_dict, os.path.join(args.log_dir, f"logs/logs_dict_{args.tag}.csv")
    )

    return model


if __name__ == "__main__":
    # for GPU memory trace
    # sys.settrace(gpu_profile)

    val_transform = [
        ToTensor(device),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    train_transform = [RandAugment(5, 30)] + val_transform

    val_transform = Compose(val_transform)
    train_transform = Compose(train_transform)

    print("train transform", train_transform)
    print("validation transform", val_transform)

    data_log_dir = os.path.join(args.log_dir, "dataset")
    if args.load_from_json:
        train_dataset = BirdClassificationRegeneration(
            transform=train_transform, regen_transform=val_transform
        )
        train_dataset.load(data_log_dir, file_name="train_cls_regen")
        val_dataset = BirdClassificationRegeneration(
            transform=val_transform, regen_transform=val_transform
        )
        val_dataset.load(data_log_dir, file_name="validation_cls_regen")
    else:
        dataset = BirdClassificationRegeneration(root_dir=args.data_dir)

        train_idx, valid_idx = train_val_split(
            dataset, p=args.train_percent, use_p_of_data=args.use_p_of_data
        )

        train_dataset = dataset.subset(
            train_idx, transform=train_transform, regen_transform=val_transform
        )
        val_dataset = dataset.subset(
            valid_idx, transform=val_transform, regen_transform=val_transform
        )

        train_dataset.save(data_log_dir, file_name="train_cls_regen")
        val_dataset.save(data_log_dir, file_name="validation_cls_regen")

        print(f"\nDataset size :     {len(dataset)}")

    print(f"Training subset:     {len(train_dataset)}")
    print(f"Validation subset:   {len(val_dataset)}")
    print(f"\nBatches per epoch: {len(train_dataset)//args.batch_size}")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=BirdClassificationRegeneration.collate_fn,
        drop_last=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=BirdClassificationRegeneration.collate_fn,
        drop_last=True,
        shuffle=False,
    )

    if args.pretrained_backend:
        print("Using transferLearning")

    model = retinanet_resnet50_fpn(
        num_classes=2,
        pretrained=args.pretrained_backend,
        pretrained_backbone=args.pretrained_backend,
        trainable_backbone_layers=5,
        extra_heads=["cls", "regen"],
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
