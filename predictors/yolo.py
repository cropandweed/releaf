import os
from numpy.typing import NDArray

from ultralytics import YOLO
from ultralytics.engine.results import Results

from .base import Predictor


class YoloPredictor(Predictor):
    """Generic YOLO predictor class that compatible with different segmentation model variants.
       The specific variant is determined by the model weights provided during initialization."""
    def __init__(self, model_path: str | None = None):
        """Initializes the predictor with specified model weights or a default model based on the predictor variant.
        Args:
            model_path (str | None): Optional path to the YOLO model weights. 
                                     If None, a default model will be loaded based on the predictor variant.
        """
        if model_path is None:
            variant = self._get_variant()
            if variant is None:
                raise ValueError('  - ERROR: Weights path must be provided for generic YOLO predictor. '
                                 'Use specific predictor variants for model training without pre-trained weights.')
            model_path = f'yolo26{variant}-seg.pt'

        self.model = YOLO(model_path)
        self._check_model_config()

        self.default_conf_threshold = 0.3
        self.default_iou_threshold = 0.7
        self.default_retina_masks = True

    def get_name(self) -> str:
        return 'yolo'

    @staticmethod
    def _get_variant() -> str | None:
        """Get the model variant based on the predictor class. This method should be overridden by specific variants."""
        return None

    def train(self, data_path: str, image_size: int, output_path: str = '.', batch_size: int = 16, epochs: int = 100):
        output = os.path.abspath(output_path)

        self.model.train(
            data=data_path,
            task='iseg',
            project=output,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            cache=True,
            amp=True,
            save=True,
        )

    def predict(self, image: NDArray, image_size: int | None = None,
                confidence: float | None = None, **kwargs) -> tuple[list[Predictor.Prediction], float]:
        iou = kwargs.get('iou', self.default_iou_threshold)
        retina_masks = kwargs.get('retina_masks', self.default_retina_masks)

        results, model_latency_ms = self._predict(image, image_size=image_size, confidence=confidence,
                                                  retina_masks=retina_masks, iou=iou)

        return ([] if results is None else [self.Prediction(r.masks.data[0].cpu().numpy(), float(r.boxes.conf[0]),
                                                            int(r.boxes.cls[0]), r.boxes.xyxy[0].tolist())
                                            for r in results[0]
                                            if r.masks is not None and r.boxes is not None]), model_latency_ms

    def _measure_latency(self, image: NDArray, confidence: float | None = None) -> float:
        return self._predict(image, image_size=max(image.shape[:2]), confidence=confidence, retina_masks=False,
                             iou=self.default_iou_threshold)[1]

    def _predict(self, image: NDArray, image_size: int | None = None, confidence: float | None = None,
                 retina_masks: bool = True, iou: float = 0.7) -> tuple[list[Results], float]:
        conf = self.default_conf_threshold if confidence is None else confidence
        start = self._timer()
        results = self.model.predict(image, verbose=False, imgsz=image_size, conf=conf,
                                     retina_masks=retina_masks, iou=iou)
        end = self._timer()
        return results, (end - start) * 1000.0

    def _check_model_config(self):
        if self.model.task != 'segment':
            raise ValueError(f'  - ERROR: Loaded model is not a segmentation model (task={self.model.task})')

        config = getattr(self.model, 'yaml', {}).get('scale', None)
        if config is not None:
            variant = self._get_variant()
            if variant is not None and not config == variant:
                print(f'  - WARNING: Loaded model variant {config} does not match predictor type {variant}')


class YoloSPredictor(YoloPredictor):
    """Small variant of YOLO segmentation model."""
    @staticmethod
    def _get_variant() -> str:
        return 's'

    def get_name(self) -> str:
        return super().get_name() + '26-small'


class YoloMPredictor(YoloPredictor):
    """Medium variant of YOLO segmentation model."""
    @staticmethod
    def _get_variant() -> str:
        return 'm'

    def get_name(self) -> str:
        return super().get_name() + '26-medium'


class YoloLPredictor(YoloPredictor):
    """Large variant of YOLO segmentation model."""
    @staticmethod
    def _get_variant() -> str:
        return 'l'

    def get_name(self) -> str:
        return super().get_name() + '26-large'
