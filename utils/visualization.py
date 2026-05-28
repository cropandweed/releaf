import cv2
import random
import numpy as np
from numpy.typing import NDArray

from predictors import Predictor

ALPHA = 0.8


def color_scheme(i: int):
    """Generates a distinct color for each prediction based on the index.
    Args:
        i (int): Index of the prediction for which to generate a color.
    Returns:
        tuple[int, int, int]: A color in BGR format to be used for visualization.
    """
    base_hue = random.randint(100, 107)
    saturation = random.randint(140, 230)
    value = random.randint(170, 255)

    hue_offset = (i * 10) % 30 - 15
    current_hue = base_hue + hue_offset
    current_hue = max(0, min(179, current_hue))

    hsv_color = np.array([[[current_hue, saturation, value]]], dtype=np.uint8)
    color_bgr_list = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0].tolist()

    color = tuple(map(int, color_bgr_list))

    return color


def visualize(image: NDArray, results: list[Predictor.Prediction], show_boxes: bool = False) -> list[str]:
    """Visualizes the predictions on the input image by overlaying colored masks and optionally drawing bounding boxes, 
       and converts them to YOLO format.
    Args:
        image (NDArray): The input image on which to visualize the predictions.
        results (list[Predictor.Prediction]): List of prediction results to visualize.
        show_boxes (bool): Whether to draw bounding boxes around the predicted objects. Defaults to False.
    Returns:
        list[str]: List of strings in YOLO format representing the predictions.
    """
    yolo_output = []

    for i, result in enumerate(sorted(results, key=lambda x: x.score, reverse=True)):
        color = color_scheme(i)

        assert image.shape[:2] == result.mask.shape[:2]

        normalized_polygon = []
        h, w = result.mask.shape[:2]
        for px, py in result.mask_polygon():
            norm_x, norm_y = px / w, py / h
            normalized_polygon.append(f'{norm_x:.6f}')
            normalized_polygon.append(f'{norm_y:.6f}')
        yolo_output.append(f'{result.class_id} ' + ' '.join(normalized_polygon))

        # Colored mask overlay + blending with original image
        mask_overlay = np.zeros_like(image, dtype=np.uint8)
        mask_overlay[result.mask] = color
        mask_bin = mask_overlay.sum(axis=2) > 0

        # Alpha blending
        blended = cv2.addWeighted(image, 1.0 - ALPHA, mask_overlay, ALPHA, 0)
        image[mask_bin] = blended[mask_bin]

        if show_boxes:
            box = result.get_box()
            if box is not None:
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)

    return yolo_output
