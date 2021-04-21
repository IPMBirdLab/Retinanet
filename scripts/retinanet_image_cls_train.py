import torch
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from retinanet.model.detection import retinanet_resnet50_fpn
from retinanet.datasets.transforms import Compose, Normalize, ToTensor, RandAugment
from retinanet.datasets.bird import BirdDetection, BirdClassification
from retinanet.datasets.utils import train_val_split, TransformDatasetWrapper
from retinanet.utils.utils import GradualWarmupScheduler
from retinanet.model.utils import outputs_to_logits, logits_to_preds
from retinanet.ops.bbox import batched_nms

from retinanet.utils.metrics import (
    calculate_metrics,
    batched_average_precision,
    MeanAveragePrecisionMeter,
)
from retinanet.utils import create_directory, Average_Meter

import os
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
parser.add_argument("--det_data_dir", default="../../dataset", type=str)
parser.add_argument("--cls_data_dir", default="../../dataset", type=str)
parser.add_argument("--load_from_json", action="store_true")
parser.add_argument("--num_workers", default=1, type=int)

###############################################################################
# Hyperparameter
###############################################################################
# detection
parser.add_argument("--det_batch_size", default=2, type=int)
parser.add_argument("--det_accumulation_steps", default=1, type=int)
parser.add_argument("--det_val_batch_size", default=None, type=int)
parser.add_argument("--det_train_percent", default=0.9, type=float)
parser.add_argument("--det_use_p_of_data", default=0.5, type=float)
# classification
parser.add_argument("--cls_batch_size", default=2, type=int)
parser.add_argument("--cls_accumulation_steps", default=1, type=int)
parser.add_argument("--cls_val_batch_size", default=None, type=int)
parser.add_argument("--cls_train_percent", default=0.9, type=float)
parser.add_argument("--cls_use_p_of_data", default=0.5, type=float)
# common
parser.add_argument("--max_epoch", default=1, type=int)

# detection
parser.add_argument("--det_opt", default="adam", type=str)
parser.add_argument("--det_lr", default=3e-4, type=float)
parser.add_argument("--det_lr_delta", default=1e-5, type=float)
parser.add_argument("--det_weight_decay", default=1e-4, type=float)
parser.add_argument("--det_lr_warmup", default=1e-1, type=float)
# SGD
parser.add_argument("--det_momentum", default=0.9, type=float)
parser.add_argument("--det_nesterov", action="store_true")
# classification
parser.add_argument("--cls_opt", default="adam", type=str)
parser.add_argument("--cls_lr", default=3e-4, type=float)
parser.add_argument("--cls_lr_delta", default=1e-5, type=float)
parser.add_argument("--cls_weight_decay", default=1e-4, type=float)
parser.add_argument("--cls_lr_warmup", default=1e-1, type=float)
# SGD
parser.add_argument("--cls_momentum", default=0.9, type=float)
parser.add_argument("--cls_nesterov", action="store_true")

# common
parser.add_argument("--pretrained", default="", type=str)
parser.add_argument("--pretrained_backend", action="store_true")

parser.add_argument("--log_dir", default="experiments", type=str)
parser.add_argument("--tag", default="", type=str)
args = parser.parse_args()

if args.cls_val_batch_size is None:
    args.cls_val_batch_size = args.cls_batch_size
if args.det_val_batch_size is None:
    args.det_val_batch_size = args.det_batch_size
###################################################################################


def dump_results_dict(logs_dict, logs_path):
    res_dict = {}
    for key in logs_dict.keys():
        res_dict[f"{key}_train"] = [logs_dict[key][0]]
        res_dict[f"{key}_val"] = [logs_dict[key][1]]

    res = pd.DataFrame.from_dict(res_dict)
    res.to_csv(logs_path, float_format="%1.5f", index=False)


class TrainClassification:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        epochs=1,
        accumulation_steps=1,
        opt="adam",
        lr=3e-4,
        lr_delta=1e-5,
        weight_decay=1e-4,
        lr_warmup=1e-1,
        momentum=0.9,
        nesterov=False,
        log_dir="experiments",
        tag="",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.lr = lr
        self.lr_delta = lr_delta
        self.weight_decay = weight_decay
        self.lr_warmup = lr_warmup
        self.momentum = momentum
        self.nesterov = nesterov
        self.log_dir = log_dir
        self.tag = tag

        self.model = model
        self.train()

        if opt == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )

        self.total_steps = (
            self.epochs * len(self.train_loader)
        ) // self.accumulation_steps

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_steps, eta_min=self.lr * self.lr_delta
        )
        if self.lr_warmup != -1:
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, self.total_steps * self.lr_warmup, self.scheduler
            )

        self.init_logger()

    def init_logger(self):
        self.train_writer = SummaryWriter(
            os.path.join(self.log_dir, f"logs/tensorboard/{self.tag}/train"),
            filename_suffix=self.tag,
        )
        self.val_writer = SummaryWriter(
            os.path.join(self.log_dir, f"logs/tensorboard/{self.tag}/val"),
            filename_suffix=self.tag,
        )
        self.step_meter = Average_Meter(["loss", "acc", "precision", "recall", "F1"])
        self.batch_meter = Average_Meter(["loss", "acc", "precision", "recall", "F1"])

        self.logs_dict = {
            "best_loss": [1000, 1000],
            "best_acc": [-1, -1],
            "best_f1": [-1, -1],
        }

    def one_batch(self, batch):
        if self.training:
            losses, cls_outputs = self.model(*batch)
        else:
            losses, _, cls_outputs = self.model(*batch)

        # Normalize our loss (if averaged)
        loss = losses["img_classification"] / self.accumulation_steps

        if self.training:
            loss.backward()

        return loss, cls_outputs

    def log_batch(self, batch, loss, cls_outputs):
        labels = batch[1]
        predicted = logits_to_preds(outputs_to_logits(cls_outputs))
        labels = torch.cat(
            list(map(lambda x: x["img_cls_labels"].unsqueeze(0), labels)), 0
        )
        acc, precision, recall, f1 = calculate_metrics(predicted.detach(), labels)

        self.batch_meter.add(
            {
                "loss": loss.item(),
                "acc": acc.item(),
                "precision": precision.item(),
                "recall": recall.item(),
                "F1": f1.item(),
            }
        )

    def one_step(self, data_iterator):
        if self.training:
            self.optimizer.zero_grad()

        for i in range(self.accumulation_steps):
            try:
                batch = next(data_iterator)

                loss, cls_outputs = self.one_batch(batch)
                self.log_batch(batch, loss, cls_outputs)
            except StopIteration:
                break

        if self.training:
            self.optimizer.step()
            self.scheduler.step()

    def log_step(self, epoch, i_step, iteration):
        loss, acc, precision, recall, f1 = self.batch_meter.get(clear=True)
        self.step_meter.add(
            {
                "loss": loss,
                "acc": acc,
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
        )

        if self.training:
            self.train_writer.add_scalar(
                "Train/Losses/image_classification_loss",
                loss,
                iteration,
            )

            self.train_writer.add_scalar(
                "HP/Cls/lr",
                self.optimizer.param_groups[0]["lr"],
                iteration,
            )

            print(
                "Epoch: {} | step: {} | Image Classification loss: {:1.5f}".format(
                    epoch,
                    i_step + 1,
                    float(loss),
                )
            )

    def one_epoch(self, epoch, data_loader):
        data_iterator = iter(data_loader)
        steps = 1
        if self.training:
            steps = self.accumulation_steps

        for i_step in range(len(data_loader) // steps):
            iteration = epoch * (len(data_loader) // steps) + i_step

            self.one_step(data_iterator)
            self.log_step(epoch, i_step, iteration)

    def log_epoch(self, epoch, train_logs, eval_logs):
        tcls_loss, tacc, tprecision, trecall, tf1 = train_logs
        cls_loss, acc, precision, recall, f1 = eval_logs
        self.train_writer.add_scalar(
            "Evaluate/Losses/image_classification_loss",
            tcls_loss,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Losses/image_classification_loss",
            cls_loss,
            epoch,
        )
        self.train_writer.add_scalar(
            "Evaluate/Metrics/Accuracy",
            tacc,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Metrics/Accuracy",
            acc,
            epoch,
        )
        self.train_writer.add_scalar(
            "Evaluate/Metrics/Precision",
            tprecision,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Metrics/Precision",
            precision,
            epoch,
        )
        self.train_writer.add_scalar(
            "Evaluate/Metrics/Recall",
            trecall,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Metrics/Recall",
            recall,
            epoch,
        )
        self.train_writer.add_scalar(
            "Evaluate/Metrics/F1",
            tf1,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Metrics/F1",
            f1,
            epoch,
        )

        print(
            "Evaluation -> Epoch: {} | Image Classification loss: {:1.5f} | Accuracy: {:1.5f} | F1: {:1.5f}".format(
                epoch,
                cls_loss,
                acc,
                f1,
            )
        )

        if tcls_loss < self.logs_dict["best_loss"][0]:
            self.logs_dict["best_loss"][0] = tcls_loss
        if tacc > self.logs_dict["best_acc"][0]:
            self.logs_dict["best_acc"][0] = tacc
        if tf1 > self.logs_dict["best_f1"][0]:
            self.logs_dict["best_f1"][0] = tf1
        if cls_loss < self.logs_dict["best_loss"][1]:
            self.logs_dict["best_loss"][1] = cls_loss
        if acc > self.logs_dict["best_acc"][1]:
            self.logs_dict["best_acc"][1] = acc
        if f1 > self.logs_dict["best_f1"][1]:
            self.logs_dict["best_f1"][1] = f1

        self.dump_results_dict()

    def chpt_callback(self, train_logs, eval_logs):
        tloss, tacc, _, _, tf1 = train_logs
        loss, acc, _, _, f1 = eval_logs

        if loss < self.logs_dict["best_loss"][1]:
            base_dir = os.path.join(self.log_dir, "checkpoints")
            create_directory(base_dir)
            torch.save(
                model.state_dict(),
                os.path.join(base_dir, f"best_chpt_{self.tag}_cls.pth"),
            )

    def one_epoch_cycle(self, epoch):
        # train
        self.one_epoch(epoch, self.train_loader)

        train_logs = self.step_meter.get(clear=True)

        # evaluate
        with torch.no_grad():
            self.eval()
            self.one_epoch(epoch, self.val_loader)
            self.train()

        eval_logs = self.step_meter.get(clear=True)
        self.chpt_callback(train_logs, eval_logs)
        self.log_epoch(epoch, train_logs, eval_logs)

    def dump_results_dict(self):
        dump_results_dict(
            self.logs_dict,
            os.path.join(self.log_dir, f"logs/logs_dict_{self.tag}.csv"),
        )

    def eval(self):
        self.model.eval()
        self.training = self.model.training

    def train(self):
        self.model.train()
        self.training = self.model.training


class TrainDetection:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        epochs=1,
        accumulation_steps=1,
        opt="adam",
        lr=3e-4,
        lr_delta=1e-5,
        weight_decay=1e-2,
        lr_warmup=1e-1,
        momentum=0.9,
        nesterov=False,
        log_dir="experiments",
        tag="",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        self.lr = lr
        self.lr_delta = lr_delta
        self.weight_decay = weight_decay
        self.lr_warmup = lr_warmup
        self.momentum = momentum
        self.nesterov = nesterov
        self.log_dir = log_dir
        self.tag = tag

        self.model = model
        self.train()

        if opt == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )

        self.total_steps = (
            self.epochs * len(self.train_loader)
        ) // self.accumulation_steps

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.total_steps, eta_min=self.lr * self.lr_delta
        )
        if self.lr_warmup != -1:
            self.scheduler = GradualWarmupScheduler(
                self.optimizer, self.total_steps * self.lr_warmup, self.scheduler
            )

        self.init_logger()

    def init_logger(self):
        self.train_writer = SummaryWriter(
            os.path.join(self.log_dir, f"logs/tensorboard/{self.tag}/train"),
            filename_suffix=self.tag,
        )
        self.val_writer = SummaryWriter(
            os.path.join(self.log_dir, f"logs/tensorboard/{self.tag}/val"),
            filename_suffix=self.tag,
        )
        self.step_meter = Average_Meter(["classification_loss", "bbox_regression_loss"])
        self.batch_meter = Average_Meter(
            ["classification_loss", "bbox_regression_loss"]
        )

        self.ap_meter = MeanAveragePrecisionMeter()

        self.logs_dict = {"best_loss": [1000, 1000], "best_map": [-1, -1]}

    def one_batch(self, batch):
        if self.training:
            losses, _ = self.model(*batch)
            self.model.eval()
            _, detections, _ = self.model(*batch)
            self.model.train()
            outputs = losses, detections
        else:
            losses, detections, _ = self.model(*batch)
            outputs = losses, detections

        # Normalize our loss (if averaged)
        if self.accumulation_steps > 1:
            losses["classification"] = (
                losses["classification"] / self.accumulation_steps
            )
            losses["bbox_regression"] = (
                losses["bbox_regression"] / self.accumulation_steps
            )

        loss = losses["classification"] + losses["bbox_regression"]

        if self.training:
            loss.backward()

        return outputs

    def log_batch(self, batch, outputs):
        losses, detections = outputs
        boxes, scores, labels = (
            [det["boxes"] for det in detections],
            [det["scores"] for det in detections],
            [det["labels"] for det in detections],
        )
        _, targets = batch
        gt_boxes, gt_labels = (
            [lbl["boxes"] for lbl in targets],
            [lbl["labels"] for lbl in targets],
        )

        ap_list = batched_average_precision(boxes, scores, labels, gt_boxes)
        self.ap_meter.add_average_precision_list(ap_list)

        self.batch_meter.add(
            {
                "classification_loss": losses["classification"].item(),
                "bbox_regression_loss": losses["bbox_regression"].item(),
            }
        )

    def one_step(self, data_iterator):
        if self.training:
            self.optimizer.zero_grad()

        for i in range(self.accumulation_steps):
            try:
                batch = next(data_iterator)

                outputs = self.one_batch(batch)

                self.log_batch(batch, outputs)
            except StopIteration:
                break

        if self.training:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.scheduler.step()

    def log_step(self, epoch, i_step, iteration):
        cls_loss, bbox_loss = self.batch_meter.get(clear=True)

        self.step_meter.add(
            {
                "classification_loss": cls_loss,
                "bbox_regression_loss": bbox_loss,
            }
        )

        if self.training:
            mAP = self.ap_meter.get_mAP()

            self.train_writer.add_scalar(
                "Train/Losses/classification_loss",
                cls_loss,
                iteration,
            )
            self.train_writer.add_scalar(
                "Train/Losses/bbox_regression_loss",
                bbox_loss,
                iteration,
            )
            self.train_writer.add_scalar(
                "Train/Metric/mAP",
                mAP,
                iteration,
            )

            self.train_writer.add_scalar(
                "HP/Det/lr",
                self.optimizer.param_groups[0]["lr"],
                iteration,
            )

            print(
                "Epoch: {} | batch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | mAP: {:.2f}".format(
                    epoch,
                    i_step + 1,
                    float(cls_loss),
                    float(bbox_loss),
                    float(mAP),
                )
            )

    def one_epoch(self, epoch, data_loader):
        data_iterator = iter(data_loader)
        steps = 1
        if self.training:
            steps = self.accumulation_steps

        for i_step in range(len(data_loader) // steps):
            iteration = epoch * (len(data_loader) // steps) + i_step

            self.one_step(data_iterator)
            self.log_step(epoch, i_step, iteration)

    def log_epoch(self, epoch, train_logs, eval_logs):
        loss_logs, map_logs = train_logs
        tcls_loss, tbbox_loss = loss_logs
        tmAP = map_logs

        loss_logs, map_logs = eval_logs
        cls_loss, bbox_loss = loss_logs
        mAP = map_logs

        self.train_writer.add_scalar(
            "Evaluate/Losses/classification_loss",
            tcls_loss,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Losses/classification_loss",
            cls_loss,
            epoch,
        )
        self.train_writer.add_scalar(
            "Evaluate/Losses/bbox_regression_loss",
            tbbox_loss,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Losses/bbox_regression_loss",
            bbox_loss,
            epoch,
        )
        self.train_writer.add_scalar(
            "Evaluate/Metrics/mAP",
            tmAP,
            epoch,
        )
        self.val_writer.add_scalar(
            "Evaluate/Metrics/mAP",
            mAP,
            epoch,
        )

        print(
            "Evaluation -> Epoch: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | mAP: {:.2f}".format(
                epoch, cls_loss, bbox_loss, mAP
            )
        )

        if tcls_loss + tbbox_loss < self.logs_dict["best_loss"][0]:
            self.logs_dict["best_loss"][0] = tcls_loss + tbbox_loss
        if tmAP > self.logs_dict["best_map"][0]:
            self.logs_dict["best_map"][0] = tmAP
        if cls_loss + bbox_loss < self.logs_dict["best_loss"][1]:
            self.logs_dict["best_loss"][1] = cls_loss + bbox_loss
        if mAP > self.logs_dict["best_map"][1]:
            self.logs_dict["best_map"][1] = mAP

        self.dump_results_dict()

    def chpt_callback(self, train_logs, eval_logs):
        loss_logs, map_logs = train_logs
        tcls_loss, tbbox_loss = loss_logs
        tmAP = map_logs

        loss_logs, map_logs = eval_logs
        cls_loss, bbox_loss = loss_logs
        mAP = map_logs

        if cls_loss + bbox_loss < self.logs_dict["best_loss"][1]:
            base_dir = os.path.join(self.log_dir, "checkpoints")
            create_directory(base_dir)
            torch.save(
                model.state_dict(),
                os.path.join(base_dir, f"best_chpt_{self.tag}_det.pth"),
            )

    def one_epoch_cycle(self, epoch):
        # train
        self.one_epoch(epoch, self.train_loader)

        loss_logs = self.step_meter.get(clear=True)
        map_logs = self.ap_meter.get_mAP(clear=True)
        train_logs = (loss_logs, map_logs)

        # evaluate
        with torch.no_grad():
            self.eval()
            self.one_epoch(epoch, self.val_loader)
            self.train()

        loss_logs = self.step_meter.get(clear=True)
        map_logs = self.ap_meter.get_mAP(clear=True)
        eval_logs = (loss_logs, map_logs)

        self.chpt_callback(train_logs, eval_logs)
        self.log_epoch(epoch, train_logs, eval_logs)

    def dump_results_dict(self):
        dump_results_dict(
            self.logs_dict,
            os.path.join(self.log_dir, f"logs/logs_dict_{self.tag}.csv"),
        )

    def eval(self):
        self.model.eval()
        self.training = self.model.training

    def train(self):
        self.model.train()
        self.training = self.model.training


def build_detection_dataloaders():
    train_transform = Compose(
        [
            ToTensor(device),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    data_log_dir = os.path.join(args.log_dir, "dataset")
    if args.load_from_json:
        dataset = BirdDetection()
        train_dataset = BirdDetection()
        train_dataset.load(data_log_dir, file_name="train_detection")
        val_dataset = BirdDetection()
        val_dataset.load(data_log_dir, file_name="validation_detection")
    else:
        dataset = BirdDetection(
            images_dir=os.path.join(args.det_data_dir, "data"),
            annotations_dir=os.path.join(args.det_data_dir, "ann"),
        )

        train_idx, valid_idx = train_val_split(
            dataset, p=args.det_train_percent, use_p_of_data=args.det_use_p_of_data
        )

        train_dataset = dataset.subset(train_idx)
        val_dataset = dataset.subset(valid_idx)

        train_dataset.save(data_log_dir, file_name="train_detection")
        val_dataset.save(data_log_dir, file_name="validation_detection")

        print(f"\nDataset size :   {len(dataset)}")
        print(f"Training subset:   {len(train_dataset)}")
        print(f"Validation subset: {len(val_dataset)}")

    train_dataset = TransformDatasetWrapper(train_dataset, train_transform)
    val_dataset = TransformDatasetWrapper(val_dataset, train_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.det_batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=BirdDetection.collate_fn,
        drop_last=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.det_val_batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=BirdDetection.collate_fn,
        shuffle=False,
    )

    return train_loader, val_loader


def build_classification_dataloaders():
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
        train_dataset = BirdClassification()
        train_dataset.load(data_log_dir, file_name="train_cls")
        val_dataset = BirdClassification()
        val_dataset.load(data_log_dir, file_name="validation_cls")
    else:
        dataset = BirdClassification(root_dir=args.data_dir)

        train_idx, valid_idx = train_val_split(
            dataset, p=args.cls_train_percent, use_p_of_data=args.cls_use_p_of_data
        )

        train_dataset = dataset.subset(train_idx)
        val_dataset = dataset.subset(valid_idx)

        train_dataset.save(data_log_dir, file_name="train_cls")
        val_dataset.save(data_log_dir, file_name="validation_cls")

        print(f"\nDataset size :     {len(dataset)}")

    print(f"Training subset:     {len(train_dataset)}")
    print(f"Validation subset:   {len(val_dataset)}")
    print(f"\nBatches per epoch: {len(train_dataset)//args.cls_batch_size}")

    train_dataset = TransformDatasetWrapper(train_dataset, train_transform)
    val_dataset = TransformDatasetWrapper(val_dataset, val_transform)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.cls_batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=BirdClassification.collate_fn,
        drop_last=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.cls_val_batch_size,
        num_workers=0 if device_str == "cuda" else args.num_workers,
        # pin_memory=True if device_str == "cuda" else False,
        collate_fn=BirdClassification.collate_fn,
        drop_last=True,
        shuffle=False,
    )

    return train_loader, val_loader


def get_detection_train_obj(model):
    train_loader, val_loader = build_detection_dataloaders()
    det_obj = TrainDetection(
        model,
        train_loader,
        val_loader,
        epochs=args.max_epoch,
        accumulation_steps=args.det_accumulation_steps,
        opt=args.det_opt,
        lr=args.det_lr,
        lr_delta=args.det_lr_delta,
        weight_decay=args.det_weight_decay,
        lr_warmup=args.det_lr_warmup,
        momentum=args.det_momentum,
        nesterov=args.det_nesterov,
        log_dir=args.log_dir,
        tag=args.tag + "_det",
    )

    return det_obj


def get_classification_train_obj(model):
    train_loader, val_loader = build_classification_dataloaders()
    cls_obj = TrainClassification(
        model,
        train_loader,
        val_loader,
        epochs=args.max_epoch,
        accumulation_steps=args.cls_accumulation_steps,
        opt=args.cls_opt,
        lr=args.cls_lr,
        lr_delta=args.cls_lr_delta,
        weight_decay=args.cls_weight_decay,
        lr_warmup=args.cls_lr_warmup,
        momentum=args.cls_momentum,
        nesterov=args.cls_nesterov,
        log_dir=args.log_dir,
        tag=args.tag + "_cls",
    )

    return cls_obj


def _train(model, train_obj_list, epochs):
    model.train()
    model.to(device)

    for epoch in range(epochs):
        for train_obj in train_obj_list:
            train_obj.one_epoch_cycle(epoch)

    map(lambda obj: obj.dump_results_dict(), train_obj_list)

    return model


if __name__ == "__main__":

    model = retinanet_resnet50_fpn(
        num_classes=2,
        pretrained=args.pretrained_backend,
        pretrained_backbone=args.pretrained_backend,
    )

    if args.pretrained != "":
        print(f"Using pretrained model : {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained))

    cls_train_obj = get_classification_train_obj(model)
    det_train_obj = get_detection_train_obj(model)

    model = _train(model, [cls_train_obj, det_train_obj], args.max_epoch)

    base_dir = os.path.join(args.log_dir, "checkpoints")
    create_directory(base_dir)
    torch.save(
        model.state_dict(),
        os.path.join(base_dir, f"chpt_{args.tag}.pth"),
    )
