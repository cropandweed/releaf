import tempfile
from PIL import Image
from numpy.typing import NDArray

import supervision as sv
from rfdetr.variants import RFDETRSegPreview, RFDETRSeg

from .base import Predictor


class DetrPredictor(Predictor):
    """Predictor implementation using RF-DETR's segmentation models for instance segmentation."""
    def __init__(self, model_path: str | None = None) -> None:
        """Initializes the predictor with an optional path to pre-trained model weights.
        Args:
            model_path (str | None): Optional path to pre-trained model weights. 
                                     If None, a default RF-DETR Segmentation Preview model will be used.
        """
        self.weights_path = model_path
        self.model = None
        self.default_threshold = 0.3
        self.optimized = False

    def get_name(self) -> str:
        return 'rfdetr-segpreview'

    def train(self, data_path: str, image_size: int, output_path: str = '.', batch_size: int = 16, epochs: int = 100):
        self.model = self._init_model(image_size)
        self.model.train(
            dataset_dir=data_path,
            output_dir=output_path,
            epochs=epochs,
            batch_size=batch_size,
            grad_accum_steps=8,
            num_classes=1,
            resolution=image_size
        )

    def predict(self, image: NDArray, image_size: int | None = None, confidence: float | None = None,
                **kwargs) -> tuple[list[Predictor.Prediction], float]:
        detections, model_latency_ms = self._predict(image, image_size=image_size, confidence=confidence)

        masks = getattr(detections, 'mask', None)
        scores = getattr(detections, 'confidence', None)
        boxes = getattr(detections, 'xyxy', None)
        class_ids = getattr(detections, 'class_id', None)

        results = []

        if scores is not None and masks is not None:
            for i, score in enumerate(scores):
                score = float(score)
                if len(masks) <= i:
                    continue

                cid = int(class_ids[i]) if class_ids is not None else 0
                results.append(self.Prediction(masks[i], score, cid, boxes[i].tolist() if boxes is not None else None))

        return results, model_latency_ms

    def _measure_latency(self, image: NDArray, confidence: float | None = None) -> float:
        return self._predict(image, image_size=max(image.shape[:2]), confidence=confidence)[1]

    def _predict(self, image: NDArray, image_size: int | None = None,
                 confidence: float | None = None) -> tuple[sv.Detections, float]:
        threshold = self.default_threshold if confidence is None else confidence

        self.model = self._init_model(max(image.shape[:2]) if image_size is None else image_size)
        if not self.optimized:
            self.model.optimize_for_inference()
            self.optimized = True

        start = self._timer()

        try:
            detections = self.model.predict(image, threshold=threshold, task='segment')
        except TypeError:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                Image.fromarray(image.astype('uint8')).save(tmp.name)
                detections = self.model.predict(image, threshold=threshold, task='segment')

        end = self._timer()
        model_latency_ms = (end - start) * 1000
        return detections[0] if isinstance(detections, list) else detections, model_latency_ms

    def _init_model(self, resolution: int) -> RFDETRSeg:
        if self.model is None or self.model.model.resolution != resolution:
            self.model = RFDETRSegPreview(
                num_classes=1, resolution=resolution) if self.weights_path is None else RFDETRSegPreview(
                    num_classes=1, resolution=resolution, pretrain_weights=self.weights_path)
            self.optimized = False
        return self.model
