import math
from collections import OrderedDict
import warnings
from functools import partial

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from ._utils import overwrite_eps
from ..utils import load_state_dict_from_url, load_chpt, get_map_location

from . import _utils as det_utils
from .anchor_utils import AnchorGenerator
from .transform import GeneralizedRCNNTransform
from .backbone_utils import resnet_fpn_backbone, _validate_trainable_layers
from ...ops.feature_pyramid_network import LastLevelP6P7, LastLevelP6
from ...ops import sigmoid_focal_loss
from torchvision.ops import boxes as box_ops


__all__ = ["RetinaNet", "retinanet_resnet50_fpn"]


def _sum(x: List[Tensor]) -> Tensor:
    res = x[0]
    for i in x[1:]:
        res = res + i
    return res


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(
        self,
        in_channels,
        num_anchors,
        num_classes,
        num_fpn_levels,
        extra_heads={},
    ):
        super().__init__()
        self.extra_heads = nn.ModuleDict(extra_heads) if len(extra_heads) > 0 else {}

        self.classification_head = RetinaNetClassificationHead(
            in_channels, num_anchors, num_classes
        )
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        res = {
            "classification": self.classification_head.compute_loss(
                targets, head_outputs, matched_idxs
            ),
            "bbox_regression": self.regression_head.compute_loss(
                targets, head_outputs, anchors, matched_idxs
            ),
        }

        for head in self.extra_heads.values():
            res.update(head.compute_loss(targets, head_outputs))

        return res

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        res = {
            "cls_logits": self.classification_head(x),
            "bbox_regression": self.regression_head(x),
        }
        for head in self.extra_heads.values():
            res.update(head(x))

        return res


class RetinaNetClassificationHead(nn.Module):
    """
    A classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes, prior_probability=0.01):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.constant_(layer.bias, 0)

        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # This is to fix using det_utils.Matcher.BETWEEN_THRESHOLDS in TorchScript.
        # TorchScript doesn't support class attributes.
        # https://github.com/pytorch/vision/pull/1697#issuecomment-630255584
        self.BETWEEN_THRESHOLDS = det_utils.Matcher.BETWEEN_THRESHOLDS

    def compute_loss(self, targets, head_outputs, matched_idxs):
        device = next(self.parameters()).device

        if targets[0].get("labels", None) is None:
            return torch.zeros(1).to(device)

        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Tensor
        losses = []

        cls_logits = head_outputs["cls_logits"]

        for targets_per_image, cls_logits_per_image, matched_idxs_per_image in zip(
            targets, cls_logits, matched_idxs
        ):
            # determine only the foreground
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            num_foreground = foreground_idxs_per_image.sum()

            # create the target classification
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][
                    matched_idxs_per_image[foreground_idxs_per_image]
                ],
            ] = 1.0

            # find indices for which anchors should be ignored
            valid_idxs_per_image = matched_idxs_per_image != self.BETWEEN_THRESHOLDS

            # compute the classification loss
            losses.append(
                sigmoid_focal_loss(
                    cls_logits_per_image[valid_idxs_per_image],
                    gt_classes_target[valid_idxs_per_image],
                    reduction="sum",
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / len(targets)

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_cls_logits = []

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Permute classification output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, -1, self.num_classes, H, W)
            cls_logits = cls_logits.permute(0, 3, 4, 1, 2)
            cls_logits = cls_logits.reshape(N, -1, self.num_classes)  # Size=(N, HWA, K)

            all_cls_logits.append(cls_logits)

        return torch.cat(all_cls_logits, dim=1)


class RetinaNetRegressionHead(nn.Module):
    """
    A regression head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
    }

    def __init__(self, in_channels, num_anchors):
        super().__init__()

        conv = []
        for _ in range(4):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        self.bbox_reg = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.bbox_reg.weight, std=0.01)
        torch.nn.init.zeros_(self.bbox_reg.bias)

        for layer in self.conv.children():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)
                torch.nn.init.zeros_(layer.bias)

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        device = next(self.parameters()).device

        if targets[0].get("boxes", None) is None:
            return torch.zeros(1).to(device)

        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Tensor
        losses = []

        bbox_regression = head_outputs["bbox_regression"]

        for (
            targets_per_image,
            bbox_regression_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, anchors, matched_idxs):
            # determine only the foreground indices, ignore the rest
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            num_foreground = foreground_idxs_per_image.numel()

            # select only the foreground boxes
            matched_gt_boxes_per_image = targets_per_image["boxes"][
                matched_idxs_per_image[foreground_idxs_per_image]
            ]
            bbox_regression_per_image = bbox_regression_per_image[
                foreground_idxs_per_image, :
            ]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

            # compute the regression targets
            target_regression = self.box_coder.encode_single(
                matched_gt_boxes_per_image, anchors_per_image
            )

            # compute the loss
            losses.append(
                torch.nn.functional.l1_loss(
                    bbox_regression_per_image, target_regression, size_average=False
                )
                / max(1, num_foreground)
            )

        return _sum(losses) / max(1, len(targets))

    def forward(self, x):
        # type: (List[Tensor]) -> Tensor
        all_bbox_regression = []

        for features in x:
            bbox_regression = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_regression)

            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, -1, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 3, 4, 1, 2)
            bbox_regression = bbox_regression.reshape(N, -1, 4)  # Size=(N, HWA, 4)

            all_bbox_regression.append(bbox_regression)

        return torch.cat(all_bbox_regression, dim=1)


class RetinaNetImageClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_features):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = num_features

        self.fc = nn.Linear(5 * 256, num_classes)

    def compute_loss(self, targets, head_outputs):
        device = next(self.parameters()).device

        if targets[0].get("img_cls_labels", None) is None:
            return {"img_classification": torch.zeros(1).to(device)}

        class_loss_fn = nn.MultiLabelSoftMarginLoss().to(device)

        cls_logits = head_outputs["img_classification"]
        if len(cls_logits.size()) < 2:
            cls_logits = cls_logits.unsqueeze(0)
        cls_labels = torch.cat(
            list(map(lambda x: x["img_cls_labels"].unsqueeze(0), targets)), 0
        )

        return {"img_classification": class_loss_fn(cls_logits, cls_labels.float())}

    def forward(self, x):
        cls_logits_list = []
        for i, features in enumerate(x):
            cls_logits = F.avg_pool2d(features, kernel_size=features.size()[-2:])
            cls_logits = cls_logits.squeeze().unsqueeze(1)
            cls_logits_list.append(cls_logits)

        cls_logits = torch.cat(cls_logits_list, 1).view(-1, 5 * 256)
        cls_logits = self.fc(cls_logits)

        return {"img_classification": cls_logits}


class RetinanettFCNHead(nn.Module):
    """the FCN implementation is originally from:
    https://github.com/pochih/FCN-pytorch/blob/master/python/fcn.py
    """

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn4 = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv6 = nn.ConvTranspose2d(
            256, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1
        )
        self.bn6 = nn.BatchNorm2d(256)
        self.classifier = nn.Conv2d(256, n_class, kernel_size=1)

    def compute_loss(self, targets, head_outputs):
        device = next(self.parameters()).device

        if targets[0].get("regen_labels", None) is None:
            return {"auto_encoder": torch.zeros(1).to(device)}

        loss_fn = nn.MSELoss().to(device)
        # loss_fn = lambda pr, gt: torch.sqrt((pr - gt).pow(2).mean())
        # loss_fn = partial(F.mse_loss, reduce=True)

        decoder_logits = head_outputs["auto_encoder"]
        # if len(cls_logits.size()) < 2:
        #     cls_logits = cls_logits.unsqueeze(0)
        regen_labels = torch.cat(
            list(map(lambda x: x["regen_labels"].unsqueeze(0), targets)), 0
        )

        return {"auto_encoder": loss_fn(decoder_logits, regen_labels.float())}

    def forward(self, x):
        output = x
        x5 = output[4]  # size=(N, 512, x.H/32, x.W/32)
        x4 = output[3]  # size=(N, 512, x.H/16, x.W/16)
        x3 = output[2]  # size=(N, 256, x.H/8,  x.W/8)
        x2 = output[1]  # size=(N, 128, x.H/4,  x.W/4)
        x1 = output[0]  # size=(N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))  # size=(N, 512, x.H/32, x.W/32)
        score = score + x4  # element-wise add, size=(N, 512, x.H/32, x.W/32)
        score = self.bn2(
            self.relu(self.deconv2(score))
        )  # size=(N, 256, x.H/16, x.W/16)
        score = score + x3  # element-wise add, size=(N, 256, x.H/16, x.W/16)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/8, x.W/8)
        score = score + x2  # element-wise add, size=(N, 128, x.H/8, x.W/8)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/4, x.W/4)
        score = score + x1  # element-wise add, size=(N, 64, x.H/4, x.W/4)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H/2, x.W/2)
        score = self.bn6(self.relu(self.deconv6(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)  # size=(N, n_class, x.H/1, x.W/1)

        return {"auto_encoder": score}  # size=(N, n_class, x.H/1, x.W/1)


class RetinaNet(nn.Module):
    """
    Implements RetinaNet.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or an OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (excluding the background).
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        head (nn.Module): Module run on top of the feature pyramid.
            Defaults to a module containing a classification and regression module.
        score_thresh (float): Score threshold used for postprocessing the detections.
        nms_thresh (float): NMS threshold used for postprocessing the detections.
        detections_per_img (int): Number of best detections to keep after NMS.
        fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training.
        bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training.
        topk_candidates (int): Number of best detections to keep before NMS.

    Example:

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import RetinaNet
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        >>> # RetinaNet needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the network generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(
        >>>     sizes=((32, 64, 128, 256, 512),),
        >>>     aspect_ratios=((0.5, 1.0, 2.0),)
        >>> )
        >>>
        >>> # put the pieces together inside a RetinaNet model
        >>> model = RetinaNet(backbone,
        >>>                   num_classes=2,
        >>>                   anchor_generator=anchor_generator)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
    }

    def __init__(
        self,
        backbone,
        num_classes,
        num_fpn_levels,
        # transform parameters
        min_size=256,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # Anchor parameters
        anchor_generator=None,
        head=None,
        extra_heads={},
        proposal_matcher=None,
        score_thresh=0.05,
        nms_thresh=0.5,
        detections_per_img=300,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        topk_candidates=1000,
    ):
        super().__init__()

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)"
            )
        self.backbone = backbone

        assert isinstance(anchor_generator, (AnchorGenerator, type(None)))

        if anchor_generator is None:
            anchor_sizes = tuple(
                (int(x * 0.4), int(x * 0.516), int(x * 0.656))
                for x in [32, 64, 128, 256, 512]
            )
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.anchor_generator = anchor_generator

        if head is None:
            head = RetinaNetHead(
                backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
                num_fpn_levels,
                extra_heads,
            )
        self.head = head

        if proposal_matcher is None:
            proposal_matcher = det_utils.Matcher(
                fg_iou_thresh,
                bg_iou_thresh,
                allow_low_quality_matches=True,
            )
        self.proposal_matcher = proposal_matcher

        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std
        )

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.topk_candidates = topk_candidates

        # used only on torchscript mode
        self._has_warned = False

    def compute_loss(self, targets, head_outputs, anchors):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor]) -> Dict[str, Tensor]
        if targets[0].get("labels", None) is not None:
            matched_idxs = []
            for anchors_per_image, targets_per_image in zip(anchors, targets):
                if targets_per_image["boxes"].numel() == 0:
                    matched_idxs.append(
                        torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64)
                    )
                    continue

                match_quality_matrix = box_ops.box_iou(
                    targets_per_image["boxes"], anchors_per_image
                )
                matched_idxs.append(self.proposal_matcher(match_quality_matrix))
        else:
            matched_idxs = None

        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)

    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits = head_outputs["cls_logits"]
        box_regression = head_outputs["bbox_regression"]

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image = [cl[index] for cl in class_logits]
            anchors_per_image, image_shape = anchors[index], image_shapes[index]

            image_boxes = []
            image_scores = []
            image_labels = []

            for box_regression_per_level, logits_per_level, anchors_per_level in zip(
                box_regression_per_image, logits_per_image, anchors_per_image
            ):
                num_classes = logits_per_level.shape[-1]

                # remove low scoring boxes
                scores_per_level = torch.sigmoid(logits_per_level).flatten()
                keep_idxs = scores_per_level > self.score_thresh
                scores_per_level = scores_per_level[keep_idxs]
                topk_idxs = torch.where(keep_idxs)[0]

                # keep only topk scoring predictions
                num_topk = min(self.topk_candidates, topk_idxs.size(0))
                scores_per_level, idxs = scores_per_level.topk(num_topk)
                topk_idxs = topk_idxs[idxs]

                anchor_idxs = topk_idxs // num_classes
                labels_per_level = topk_idxs % num_classes

                boxes_per_level = self.box_coder.decode_single(
                    box_regression_per_level[anchor_idxs],
                    anchors_per_level[anchor_idxs],
                )
                boxes_per_level = box_ops.clip_boxes_to_image(
                    boxes_per_level, image_shape
                )

                image_boxes.append(boxes_per_level)
                image_scores.append(scores_per_level)
                image_labels.append(labels_per_level)

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(
                image_boxes, image_scores, image_labels, self.nms_thresh
            )
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )

        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if targets is not None and targets[0].get("boxes", None) is not None:
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            "Expected target boxes to be a tensor"
                            "of shape [N, 4], got {:}.".format(boxes.shape)
                        )
                else:
                    raise ValueError(
                        "Expected target boxes to be of type "
                        "Tensor, got {:}.".format(type(boxes))
                    )

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        if targets is not None and targets[0].get("boxes", None) is not None:
            images, targets = self.transform(images, targets)
        else:
            images, _ = self.transform(images, None)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None and targets[0].get("boxes", None) is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        " Found invalid box {} for target at index {}.".format(
                            degen_bb, target_idx
                        )
                    )

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if targets is not None:
            # compute the losses
            losses = self.compute_loss(targets, head_outputs, anchors)

        if not self.training:
            # recover level sizes
            num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
            HW = 0
            for v in num_anchors_per_level:
                HW += v
            HWA = head_outputs["cls_logits"].size(1)
            A = HWA // HW
            num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

            # split outputs per level
            split_head_outputs: Dict[str, List[Tensor]] = {}
            for k in head_outputs:
                if k not in ["img_classification", "auto_encoder"]:
                    split_head_outputs[k] = list(
                        head_outputs[k].split(num_anchors_per_level, dim=1)
                    )
            split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

            # compute the detections
            detections = self.postprocess_detections(
                split_head_outputs, split_anchors, images.image_sizes
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn(
                    "RetinaNet always returns a (Losses, Detections) tuple in scripting"
                )
                self._has_warned = True
            return losses, detections

        # type: (Dict[str, Tensor],
        #        List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor],
        #        List[Dict[str, Tensor]]]
        # TODO: returning a dynamyc component statically is wrong.
        #      head outputs must be returned and used dynamically.
        if self.training:
            return losses, head_outputs.get("img_classification", None)
        if targets is not None:
            return losses, detections, head_outputs.get("img_classification", None)
        return detections


model_urls = {
    "retinanet_resnet50_fpn_coco": "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
}


def retinanet_resnet50_fpn(
    pretrained=False,
    progress=True,
    num_classes=91,
    pretrained_backbone=True,
    trainable_backbone_layers=None,
    extra_heads=[],
    **kwargs,
):
    """
    Constructs a RetinaNet model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with values
          between ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Example::

        >>> model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        extra_heads (list[str]): list of extra heads to use. possible choices are ["cls", "regen"].
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    # skip P2 because it generates too many anchors (according to their paper)
    backbone = resnet_fpn_backbone(
        "resnet50",
        pretrained_backbone,
        returned_layers=[1, 2, 3, 4],
        extra_blocks=LastLevelP6(256, 256),
        trainable_layers=trainable_backbone_layers,
    )
    num_fpn_levels = 5

    extra_head_objs = {}
    if "cls" in extra_heads:
        print("***** adding Image classification head")
        image_classification_head = RetinaNetImageClassificationHead(
            backbone.out_channels, num_classes, num_fpn_levels
        )
        extra_head_objs.update({"image_classification_head": image_classification_head})
    if "regen" in extra_heads:
        print("***** adding Image Regeneration head")
        autoencoder_head = RetinanettFCNHead(
            n_class=3
        )  # 3 number of output channels for RGB
        extra_head_objs.update({"autoencoder_head": autoencoder_head})

    model = RetinaNet(
        backbone, num_classes, num_fpn_levels, extra_heads=extra_head_objs, **kwargs
    )
    if pretrained:
        # state_dict = load_state_dict_from_url(
        #     model_urls["retinanet_resnet50_fpn_coco"], progress=progress
        # )
        state_dict = load_state_dict_from_url(
            model_urls["retinanet_resnet50_fpn_coco"],
            progress=progress,
            map_location=get_map_location(),
        )
        model = load_chpt(model, state_dict)
        overwrite_eps(model, 0.0)
    return model
