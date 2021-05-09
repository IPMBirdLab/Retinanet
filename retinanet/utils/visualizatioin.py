import torch
from torch import nn
import torchvision
from retinanet.model.detection.transform import GeneralizedRCNNTransform

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from VideoToolkit.tools import rescal_to_image, get_cv_resize_function
from skimage.transform import resize
from retinanet.utils import create_directory


def get_features(model, images, device=None):
    transform = GeneralizedRCNNTransform(
        800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    images, _ = transform(images, None)

    # get the features from the backbone
    features = model.backbone(images.tensors.to(device))

    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())
    return features


def vis_features(
    model,
    img,
    gt_boxes=None,
    vis_boxes=True,
    threshold=0,
    path=None,
    device=None,
    thickness=2,
):
    model.eval()
    # get features
    features = get_features(model, [img], device)
    features = [feat.mean(1) for feat in features]

    imact = [feat.squeeze().cpu().detach().numpy() for feat in features]

    for im in imact:
        print(im.shape)

    # get predictions
    predicted = model([img])
    keep = torchvision.ops.nms(predicted[0]["boxes"], predicted[0]["scores"], 0.1)
    keep = keep.cpu().numpy()
    boxes = list(np.floor(predicted[0]["boxes"].cpu().detach().numpy()[keep]))
    scores = list(predicted[0]["scores"].cpu().detach().numpy()[keep])

    # Visualize

    # reverting the transformation done by datagenerator
    img_n = img.cpu().permute((1, 2, 0)).numpy().copy() * np.array(
        [0.229, 0.224, 0.225]
    ) + np.array([0.485, 0.456, 0.406])

    # visualize groundtruth boxes
    if gt_boxes is not None:
        for box in gt_boxes:
            cv2.rectangle(
                img_n,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                thickness,
            )

    # visualize boxes
    if vis_boxes:
        for box, score in zip(boxes, scores):
            if score > threshold:
                cv2.rectangle(
                    img_n,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0),
                    thickness,
                )

    # plotting
    img_n = resize(img_n, (400, 600), anti_aliasing=True)
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, axarr = plt.subplots(
        2, 3, figsize=(3 * img_n.shape[1] * px, 2 * img_n.shape[0] * px)
    )

    axarr[0, 0].imshow(img_n)
    # visualize features
    for j in range(1, 6):
        axarr[j // 3, j % 3].imshow(resize(imact[j - 1], img_n.shape[:2]))

    if path is not None:
        directory = os.path.split(path)[0]
        create_directory(directory)
        fig.savefig(path, format="jpg")
        cv2.imwrite(
            path[:-4] + "_img.jpg",
            cv2.cvtColor(np.float32(img_n) * 255, cv2.COLOR_RGB2BGR),
        )
    return fig


def vis_features_CAM(
    model,
    img,
    gt_boxes=None,
    vis_boxes=True,
    threshold=0,
    path=None,
    device=None,
    thickness=2,
):
    model.eval()
    # get features
    features = get_features(model, [img], device)

    weights = nn.Parameter(
        model.head.extra_heads.image_classification_head.fc.weight.t().unsqueeze(0)
    )

    get_weight = lambda weight, idx, cls: weight[:, :, cls].view(5, 256)[idx, :][
        None, :, None, None
    ]

    features = [
        (feat * get_weight(weights, i, 1)).mean(1) for i, feat in enumerate(features)
    ]

    imact = [feat.squeeze().cpu().detach().numpy() for feat in features]

    for im in imact:
        print(im.shape)

    # get predictions
    predicted = model([img])
    keep = torchvision.ops.nms(predicted[0]["boxes"], predicted[0]["scores"], 0.1)
    keep = keep.cpu().numpy()
    boxes = list(np.floor(predicted[0]["boxes"].cpu().detach().numpy()[keep]))
    scores = list(predicted[0]["scores"].cpu().detach().numpy()[keep])

    # Visualize
    # reverting the transformation done by datagenerator
    img_n = img.cpu().permute((1, 2, 0)).numpy().copy() * np.array(
        [0.229, 0.224, 0.225]
    ) + np.array([0.485, 0.456, 0.406])

    # visualize groundtruth boxes
    if gt_boxes is not None:
        for box in gt_boxes:
            cv2.rectangle(
                img_n,
                (int(box[0]), int(box[1])),
                (int(box[2]), int(box[3])),
                (0, 255, 0),
                thickness,
            )

    # visualize boxes
    if vis_boxes:
        for box, score in zip(boxes, scores):
            if score > threshold:
                cv2.rectangle(
                    img_n,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0),
                    thickness,
                )
    # plotting
    img_n = resize(img_n, (400, 600), anti_aliasing=True)
    px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
    fig, axarr = plt.subplots(
        2, 3, figsize=(3 * img_n.shape[1] * px, 2 * img_n.shape[0] * px)
    )

    axarr[0, 0].imshow(img_n)
    # visualize features
    for j in range(1, 6):
        axarr[j // 3, j % 3].imshow(resize(imact[j - 1], img_n.shape[:2]))

    if path is not None:
        directory = os.path.split(path)[0]
        create_directory(directory)
        fig.savefig(path, format="jpg")
        cv2.imwrite(
            path[:-4] + "_img.jpg",
            cv2.cvtColor(np.float32(img_n) * 255, cv2.COLOR_RGB2BGR),
        )
    return fig
