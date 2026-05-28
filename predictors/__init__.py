from .base import Predictor

try:
    from .yolo import YoloPredictor, YoloSPredictor, YoloMPredictor, YoloLPredictor
except ImportError:
    YoloPredictor = None
    YoloSPredictor = None
    YoloMPredictor = None
    YoloLPredictor = None

try:
    from .detectron import DetectronR50Predictor, DetectronR101Predictor, DetectronResnextPredictor
except ImportError:
    DetectronR50Predictor = None
    DetectronR101Predictor = None
    DetectronResnextPredictor = None

try:
    from .detr import DetrPredictor
except ImportError:
    DetrPredictor = None


PREDICTORS = {
    'yolo': YoloPredictor,
    'yolo-s': YoloSPredictor,
    'yolo-m': YoloMPredictor,
    'yolo-l': YoloLPredictor,
    'detectron-r50': DetectronR50Predictor,
    'detectron-r101': DetectronR101Predictor,
    'detectron-resnext': DetectronResnextPredictor,
    'detr': DetrPredictor,
}


def create_predictor(predictor_name: str, weights_path: str | None = None) -> Predictor:
    """Factory function to create predictor instances based on the specified architecture name.
    Args:
        predictor_name (str): Name of the predictor architecture.
        weights_path (str | None): Optional path to model weights. If None, default weights will be used.
    Returns:
        Predictor: An instance of the specified predictor.
    Raises:
        ValueError: If predictor name is not supported.
        ImportError: If predictor class is not available due to missing dependencies.
    """
    name = predictor_name.lower()
    if name not in PREDICTORS:
        raise ValueError(
            f'  - ERROR: Predictor "{predictor_name}" is not supported. Available options: {list(PREDICTORS.keys())}')

    predictor_cls = PREDICTORS[name]
    if predictor_cls is None:
        raise ImportError(f'  - ERROR: Predictor "{predictor_name}" is not available.')

    return predictor_cls(weights_path)
