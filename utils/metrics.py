from abc import ABCMeta, abstractmethod
from typing import Any, TypeVar, Generic
import numpy as np
from numpy.typing import NDArray
from pycocotools import mask as mask_utils


def binary_mask_to_rle(binary_mask: NDArray) -> Any:
    """Converts a binary mask to COCO RLE format using pycocotools.
    Args:
        binary_mask (NDArray): 2D binary mask where 1 represents the object and 0 represents the background.
    Returns:
        dict: RLE-encoded mask in COCO format.
    """
    if not binary_mask.flags['F_CONTIGUOUS']:
        binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(binary_mask.astype(np.uint8))
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def dice(mask1: NDArray, mask2: NDArray) -> float:
    """Calculates the Dice coefficient between two binary masks.
    Args:
        mask1 (NDArray): First binary mask.
        mask2 (NDArray): Second binary mask.
    Returns:
        float: The Dice coefficient.
    """
    inter = np.logical_and(mask1, mask2).sum()
    s1 = mask1.sum()
    s2 = mask2.sum()
    if s1 + s2 == 0:
        return 1.0
    return 2.0 * inter / (s1 + s2)


def symmetric_best_dice(gt_masks: list[NDArray], pred_masks: list[NDArray]) -> float:
    """Calculates the Symmetric Best Dice (SBD) metric between two sets of binary masks.
    Args:
        gt_masks (list[NDArray]): List of ground-truth binary masks.
        pred_masks (list[NDArray]): List of predicted binary masks.
    Returns:
        float: The Symmetric Best Dice (SBD) score.
    """
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    dice_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
    for i, g in enumerate(gt_masks):
        for j, p in enumerate(pred_masks):
            dice_mat[i, j] = dice(g, p)

    best_dice_gt = dice_mat.max(axis=1)
    best_dice_pred = dice_mat.max(axis=0)
    return float(0.5 * (best_dice_gt.mean() + best_dice_pred.mean()))


T = TypeVar('T')
class Stats(Generic[T], metaclass=ABCMeta):
    """Generic statistics class for evaluating predictions against ground truth using a specified IoU threshold 
       and accumulating true positives, false positives, and false negatives to calculate performance metrics."""
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, gt: list[T], pred: list[T], iou_thresh=0.5):
        """Adds a batch of ground truth and predicted items to the statistics, 
           calculating true positives, false positives, and false negatives based on the specified IoU threshold.
        Args:            
            gt (list[T]): List of ground-truth items.
            pred (list[T]): List of predicted items.
            iou_thresh (float): IoU threshold to determine matches between ground truth and predictions.
        """
        n_gt = len(gt)
        n_pred = len(pred)
        if n_gt == 0 and n_pred == 0:
            return
        if n_gt == 0:
            self.fp += n_pred
        elif n_pred == 0:
            self.fn += n_gt
        else:
            iou_mat = np.zeros((n_gt, n_pred))
            for i, g in enumerate(gt):
                for j, p in enumerate(pred):
                    iou_mat[i, j] = self._iou(g, p)

            tp = 0
            iou_w = iou_mat.copy()
            while True:
                idx = np.unravel_index(np.argmax(iou_w), iou_w.shape)
                if iou_w[idx] < iou_thresh:
                    break
                tp += 1
                iou_w[idx[0], :] = -1
                iou_w[:, idx[1]] = -1
            self.tp += tp
            self.fp += n_pred - tp
            self.fn += n_gt - tp

    @staticmethod
    @abstractmethod
    def _iou(a: T, b: T) -> float:
        """Calculates the Intersection over Union (IoU) between two items of type T. This method must be implemented 
           by subclasses to define how IoU is computed for the specific data type being evaluated.
        Args:
            a (T): First item.
            b (T): Second item.
        Returns:
            float: The IoU between the two items.
        """

    def __str__(self) -> str:
        """Returns a string representation of the precision, recall, and F1 score based on the accumulated statistics.
        Returns:            
            str: Formatted string showing precision, recall, F1 score, and counts of TP, FP, FN.
        """
        p = self.tp / (self.tp + self.fp + 1e-6)
        r = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * p * r / (p + r + 1e-6)
        return f'P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f} (TP:{self.tp}, FP:{self.fp}, FN:{self.fn})'


class MaskStats(Stats[NDArray]):
    """Statistics class for evaluating binary mask predictions against ground-truth masks using IoU for matching."""
    @staticmethod
    def _iou(a: NDArray, b: NDArray):
        inter = np.logical_and(a, b).sum()
        union = np.logical_or(a, b).sum()
        return inter / union if union > 0 else 0


class BoxStats(Stats[list[float]]):
    """Statistics class for evaluating bounding box predictions against ground-truth boxes using IoU for matching."""
    @staticmethod
    def _iou(a: list[float], b: list[float]):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (a[2] - a[0]) * (a[3] - a[1])
        area2 = (b[2] - b[0]) * (b[3] - b[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0
