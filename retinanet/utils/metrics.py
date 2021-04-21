import torch
import torchvision
import numpy as np


# Detection metrics
def _match_predictions(boxes, gt_boxes):
    match_matrix = torchvision.ops.box_iou(boxes, gt_boxes)
    matched_pred_scores, _ = match_matrix.max(dim=1)
    return matched_pred_scores


def batched_average_precision(boxes, scores, labels, gt_boxes):
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
            ap = torch.zeros(1).to(boxes_per_img.device)
        else:
            keep = torchvision.ops.batched_nms(
                boxes_per_img, scores_per_img, labels_per_img, 0.5
            )
            boxes_per_img, scores_per_img = (
                boxes_per_img[keep],
                scores_per_img[keep],
            )

            match_scores = _match_predictions(boxes_per_img, gt_boxes_per_img)
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
        ap_list.append(ap.item())
    return ap_list


class MeanAveragePrecisionMeter(object):
    def __init__(self):
        self._reinitialize()

    def _reinitialize(self):
        self.ap_list = None

    def add_average_precision_list(self, ap_list):
        if not isinstance(ap_list, list) and not isinstance(ap_list, np.ndarray):
            raise TypeError

        if isinstance(ap_list, list):
            ap_list = np.array(ap_list)

        if self.ap_list is None:
            self.ap_list = ap_list
        else:
            self.ap_list = np.concatenate((self.ap_list, ap_list))

    def get_mAP(self, clear=False):
        res = np.sum(self.ap_list) / self.ap_list.shape[0]
        if clear:
            self._reinitialize()

        return res


# Classification metrics
def accuracy(preds, labels):
    true_preds = (preds == labels).float().sum(0)
    total = len(labels)

    acc = true_preds / total
    return acc


def precision_recall(preds, labels):
    true_positive = ((preds == labels) * labels).float().sum(0)
    total_predicted_positive = (preds == 1).float().sum(0)
    total_actual_positive = (labels == 1).float().sum(0)

    percision = true_positive / torch.maximum(
        total_predicted_positive,
        torch.tensor([1, 1]).to(total_predicted_positive.device),
    )
    recall = true_positive / torch.maximum(
        total_actual_positive, torch.tensor([1, 1]).to(total_actual_positive.device)
    )

    return percision, recall


def _f1(percision, recall):
    f1 = (2 * percision * recall) / torch.maximum(
        (percision + recall), torch.tensor([1, 1]).to(percision.device)
    )
    return f1


def f1(preds, labels):
    percision, recall = precision_recall(preds, labels)
    return _f1(percision, recall)


def calculate_metrics(preds, labels):
    acc = accuracy(preds, labels)
    percision, recall = precision_recall(preds, labels)
    f1 = _f1(percision, recall)

    return acc[1], percision[1], recall[1], f1[1]