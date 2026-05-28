from abc import ABCMeta, abstractmethod
import time
import torch
import cv2
import numpy as np
from numpy.typing import NDArray


class Predictor(metaclass=ABCMeta):
    """Abstract base class for all predictors defining interfaces for training and prediction."""

    class Prediction:
        """Data structure to hold individual prediction results, including mask, confidence score, class ID, 
           and optional bounding box."""
        def __init__(self, mask: NDArray, score: float, class_id: int, box: list[float] | None = None):
            """Initialization.
            Args:
                mask (NDArray): Binary mask of the predicted object.
                score (float): Confidence score of the prediction.
                class_id (int): Class ID of the predicted object.
                box (list[float] | None): Optional bounding box in [x_min, y_min, x_max, y_max] format, if available.
            """
            self.mask = mask.astype(bool)
            self.score = score
            self.class_id = class_id
            self.box = box

        def get_box(self) -> list[float]:
            """Get bounding box for the prediction. If no box is available, it will be computed from the mask.
            Returns:
                list[float]: Bounding box in [x_min, y_min, x_max, y_max] format.
            """
            if self.box is not None:
                return self.box

            mask = self.mask_polygon()
            x_coords = [p[0] for p in mask]
            y_coords = [p[1] for p in mask]
            return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

        def mask_polygon(self) -> list[tuple[float, float]]:
            """Convert the binary mask to a polygon representation by extracting contours.
            Returns:
                list[tuple[float, float]]: List of (x, y) coordinates representing the polygon vertices.
            """
            mask_bool = (np.asarray(self.mask) > 0.5).astype('uint8')
            contours = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(contours) == 0:
                raise ValueError('  - ERROR: invalid mask for prediction')

            return [(p[0], p[1]) for p in max(contours, key=cv2.contourArea).reshape(-1, 2).astype(int)]

    def get_name(self) -> str:
        """Get the name of the predictor architecture for logging purposes.
        Returns:
            str: Name of the predictor architecture.
        """
        return 'undefined'

    @abstractmethod
    def train(self, data_path: str, image_size: int, output_path: str = '.',
              batch_size: int = 16, epochs: int = 100):
        """Train the model on the provided dataset. The implementation should handle dataset loading, training loop, 
           and saving the trained model.
        Args:
            data_path (str): Path to the training dataset in format corresponding to the respective architecture.
            image_size (int): Size to which input images should be resized during training.
            output_path (str): Directory where the trained model and logs will be saved.
            batch_size (int): Number of samples per training batch.
            epochs (int): Number of training epochs."""

    @abstractmethod
    def predict(self, image: NDArray, image_size: int | None = None,
                confidence: float | None = None, **kwargs) -> tuple[list[Prediction], float]:
        """Apply model inference to the input image and return a list of Prediction objects and inference latency.
        Args:
            image (NDArray): Input image for prediction.
            image_size (int | None): Optional size to which the input image should be resized for inference. 
                                     If None, architecture defaults are applied.
            confidence (float | None): Optional confidence threshold for filtering predictions. 
                                       If None, architecture defaults are applied.
            **kwargs: Additional keyword arguments specific to the predictor implementation.
        Returns:
            tuple: (predictions, latency)
                - predictions (list[Prediction]): List of Prediction objects computed for the input image.
                - latency (float): Inference latency in milliseconds.
        """

    def measure_latency(self, imgsz: int, runs: int = 100, warmup: int = 100,
                        confidence: float | None = None) -> float:
        """Measure the average inference latency of the predictor on a dummy image of specified size.
        Args:
            imgsz (int): Size of the dummy input image (imgsz x imgsz).
            runs (int): Number of inference runs to average over for latency measurement.
            warmup (int): Number of warmup runs to stabilize GPU performance before measurement.
            confidence (float | None): Optional confidence threshold to use during latency measurement. 
                                       If None, architecture defaults are applied.
        Returns:
            float: Average inference latency in milliseconds.
        """
        dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

        # Warmup phase to stabilize GPU performance
        for _ in range(warmup):
            self._measure_latency(dummy_img, confidence)

        # Measure inference speed
        total = 0.0
        for _ in range(runs):
            total += self._measure_latency(dummy_img, confidence)

        return total / runs

    def _measure_latency(self, image: NDArray, confidence: float | None = None) -> float:
        """Internal method to measure latency for a single inference run without converting results or performing 
           any additional processing. This method should be implemented by each predictor to perform a single prediction 
           with optimized performance.
        Args:            
            image (NDArray): Input image for prediction.
            confidence (float | None): Optional confidence threshold for filtering predictions. 
                                       If None, architecture defaults are applied.
        Returns:
            float: Inference latency in milliseconds.
        """
        return self.predict(image, image_size=max(image.shape[:2]), confidence=confidence)[1]

    @staticmethod
    def _timer():
        """Utility method to synchronize CUDA operations and return the current time for accurate latency measurement."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()
