import torch
import torchvision


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