from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from timeit import default_timer as timer

def compute_percentile(
        input: torch.Tensor,
        percentile: float
) -> torch.Tensor:
    """
    Computes the percentile over a batch
    :param input: input tensor B x l
    :param percentile: The percentile to compute
    """

    return input.kthvalue(1 + round(.01 * float(percentile) * (input.size(-1) - 1)), dim=-1).values

def label_to_contour(img: torch.Tensor) -> torch.Tensor:
    """
    Extracts edges from labels.
    :param img: Input image containing labels, expected input of shape (N,c,w,d[,h])
    """
    channels = img.shape[1]
    if img.ndimension() == 4:
        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=img.device)
        kernel = kernel.repeat(channels, 1, 1, 1)
        contour_img = F.conv2d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
    elif img.ndimension() == 5:
        kernel = -1 * torch.ones(3, 3, 3, dtype=torch.float32, device=img.device)
        kernel[1, 1, 1] = 26
        kernel = kernel.repeat(channels, 1, 1, 1, 1)
        contour_img = F.conv3d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
    else:
        raise ValueError(f"Unsupported img dimension: {img.ndimension()}, available options are [4, 5].")

    contour_img.clamp_(min=0.0, max=1.0)
    return contour_img

def compute_mean_distances(
        seg_pred: Union[np.ndarray, torch.Tensor],
        seg_gt: Union[np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    Compute the mean surface distances.

    Args:
        seg_pred: the predicted binary or labelfield image.
        seg_gt: the actual binary or labelfield image.
    """
    device = seg_pred.device

    # start = timer()
    edges_pred = label_to_contour(seg_pred)
    edges_gt = label_to_contour(seg_gt)
    # end = timer()
    # print("Extract edges: {}".format(end-start))

    surface_distance = get_surface_distances(edges_pred, edges_gt, 2000, 5000)

    return surface_distance.mean(dim=-1)


def compute_hausdorff_distances(
        seg_pred: Union[np.ndarray, torch.Tensor],
        seg_gt: Union[np.ndarray, torch.Tensor],
        percentile: Optional[float] = None,
        directed: bool = False,
) -> torch.Tensor:
    """
    Compute the Hausdorff distance. The user has the option to calculate the
    directed or non-directed Hausdorff distance. By default, the non-directed
    Hausdorff distance is calculated. In addition, specify the `percentile`
    parameter can get the percentile of the distance.

    Args:
        seg_pred: the predicted binary or labelfield image.
        seg_gt: the actual binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: calculate directed Hausdorff distance. Defaults to ``False``.
    """

    device = seg_pred.device

    # start = timer()
    edges_pred = label_to_contour(seg_pred)
    edges_gt = label_to_contour(seg_gt)
    # end = timer()
    # print("Extract edges: {}".format(end-start))

    hd = compute_percent_hausdorff_distance(edges_pred, edges_gt, percentile)
    if directed:
        return hd

    hd = torch.max(hd, compute_percent_hausdorff_distance(edges_pred, edges_gt, percentile))

    # end = timer()
    # print("Total time: {}".format(end - start))
    return hd


def compute_percent_hausdorff_distance(
        edges_pred: torch.Tensor,
        edges_gt: torch.Tensor,
        percentile: Optional[float] = None,
):
    """
    This function is used to compute the directed Hausdorff distance.

    Args:
        edges_pred: the edge of the predictions.
        edges_gt: the edge of the ground truth.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
    """
    # start = timer()
    surface_distance = get_surface_distances(edges_pred, edges_gt, 2000, 5000)
    # end = timer()
    # print("Surf dists: {}".format(end - start))

    # for input without foreground
    # if surface_distance.shape == (0,):
    #     return np.inf

    if not percentile:
        return surface_distance.max(dim=-1)[0]
    elif 0 <= percentile <= 100:
        return compute_percentile(surface_distance, percentile)
    else:
        raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")


def get_surface_distances(
        edges_pred: torch.Tensor,
        edges_gt: torch.Tensor,
        num_samples_pred: Optional[int] = 5000,
        num_samples_gt: Optional[int] = 5000,
        dtype: Optional[torch.dtype] = torch.float32
) -> torch.Tensor:
    """
    This function is used to compute the surface distances from `seg_pred` to `seg_gt`.

    Args:
        edges_pred: the edge of the predictions.
        edges_gt: the edge of the ground truth.
        num_samples_pred: Number of samples to use from prediction.
        num_samples_gt: Number of samples to use from ground-truth.
        dtype: The dtype to use for computation
    """

    # if not torch.any(edges_pred):
    #     return torch.zeros((0), device=edges_gt.device)
    #
    # if not np.any(edges_gt):
    #     return torch.tensor([float('Inf')], device=edges_gt.device)

    # TODO convert IJK to RAS using meta inf if available

    ep = torch.zeros((edges_pred.shape[0], num_samples_pred, 3),
                     device=edges_pred.device, dtype=dtype, requires_grad=True)
    eg = torch.zeros((edges_pred.shape[0], num_samples_gt, 3),
                     device=edges_pred.device, dtype=dtype, requires_grad=True)

    for b in range(ep.shape[0]):
        nzp = torch.nonzero(edges_pred[b, 0, :], as_tuple=False).float()
        nzg = torch.nonzero(edges_gt[b, 0, :], as_tuple=False).float()

        pred_samp = min(nzp.shape[0], num_samples_pred)
        gt_samp = min(nzg.shape[0], num_samples_gt)

        pidx = torch.randperm(nzp.shape[0])[:pred_samp]
        gidx = torch.randperm(nzg.shape[0])[:gt_samp]

        ep[b, :pred_samp] = nzp[pidx]
        eg[b, :gt_samp] = nzg[gidx]
        # Pad rest of row with first value if too few samples
        ep[b, pred_samp:] = ep[b, 0]
        eg[b, gt_samp:] = eg[b, 0]

    # Compute distances between each pair of edge indices
    dis = torch.cdist(ep, eg)

    # Take smallest distance for each row to get distance to closest point
    surface_distance, _ = dis.min(dim=-1)
    return surface_distance



