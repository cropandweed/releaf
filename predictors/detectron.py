import os
import torch
from numpy.typing import NDArray

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from .base import Predictor


class DetectronPredictor(Predictor):
    """Predictor implementation using Detectron2's Mask R-CNN architecture for instance segmentation. The specific model 
       variant can be selected by providing a configuration file from the Detectron2 model zoo during initialization."""
    class ValidationTrainer(DefaultTrainer):
        """Custom trainer class that extends Detectron2's DefaultTrainer to include COCO evaluation during training."""
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    def __init__(self, config_path: str, weights_path: str | None = None):
        """Initializes the predictor with a specific model configuration and optional pre-trained weights.
        Args:
            config_path (str): Path to the Detectron2 model configuration file (yaml) from the model zoo.
            weights_path (str | None): Optional path to pre-trained model weights. 
                                       If None, the default weights from the model zoo will be used.
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_path))
        self.cfg.MODEL.WEIGHTS = weights_path if weights_path is not None else model_zoo.get_checkpoint_url(config_path)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.default_score_thresh = 0.3
        self.predictor = None

    def get_name(self) -> str:
        return 'detectron2'

    def train(self, data_path: str, image_size: int, output_path: str = '.', batch_size: int = 16, epochs: int = 100):
        data_train = self._register_datasets(data_path, 'train')
        data_val = self._register_datasets(data_path, 'valid')

        self.cfg.DATASETS.TRAIN = (data_train,)
        self.cfg.DATASETS.TEST = (data_val,)
        self.cfg.OUTPUT_DIR = os.path.abspath(output_path)

        self.cfg.INPUT.MIN_SIZE_TRAIN = (image_size,)
        self.cfg.INPUT.MAX_SIZE_TRAIN = image_size

        self._set_size(image_size)

        # Hyperparameter configuration
        self.cfg.DATALOADER.NUM_WORKERS = 4
        self.cfg.SOLVER.IMS_PER_BATCH = batch_size
        self.cfg.SOLVER.BASE_LR = batch_size * 0.00125  # Learning rate

        iters_per_epoch = max(1, len(DatasetCatalog.get(data_train)) // batch_size)
        max_iters = iters_per_epoch * epochs

        # Learning rate steps
        self.cfg.SOLVER.MAX_ITER = max_iters
        self.cfg.SOLVER.STEPS = (int(max_iters * 0.7), int(max_iters * 0.9))
        self.cfg.SOLVER.GAMMA = 0.1

        self.cfg.SOLVER.CHECKPOINT_PERIOD = iters_per_epoch
        self.cfg.TEST.EVAL_PERIOD = iters_per_epoch

        trainer = self.ValidationTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    def predict(self, image: NDArray, image_size: int | None = None,
                confidence: float | None = None, **kwargs) -> tuple[list[Predictor.Prediction], float]:
        outputs, model_latency_ms = self._predict(image, image_size=image_size, confidence=confidence)

        results = []
        instances = outputs['instances'].to('cpu')
        if instances.has('pred_masks') and instances.has('scores') and instances.has('pred_classes'):
            masks = instances.pred_masks.numpy()
            boxes = list(getattr(instances, 'pred_boxes')) if instances.has('pred_boxes') else None

            results = [self.Prediction(mask, float(score), int(class_id),
                                       boxes[i].tolist() if boxes is not None else None)
                       for i, (mask, score, class_id) in enumerate(zip(masks, instances.scores.numpy(),
                                                                       instances.pred_classes.numpy()))
                       if mask is not None]

        return results, model_latency_ms

    def _measure_latency(self, image: NDArray, confidence: float | None = None) -> float:
        return self._predict(image, image_size=max(image.shape[:2]), confidence=confidence)[1]

    def _predict(self, image: NDArray, image_size: int | None = None,
                 confidence: float | None = None) -> tuple[dict, float]:
        thresh = self.default_score_thresh if confidence is None else confidence
        if thresh != self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
            self.predictor = None

        if image_size is not None and (
                self.cfg.INPUT.MIN_SIZE_TEST != image_size or self.cfg.INPUT.MAX_SIZE_TEST != image_size):
            self._set_size(image_size)
            self.predictor = None

        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)

        start = self._timer()
        outputs = self.predictor(image)
        end = self._timer()
        return outputs, (end - start) * 1000.0

    def _set_size(self, image_size: int):
        self.cfg.INPUT.MIN_SIZE_TEST = image_size
        self.cfg.INPUT.MAX_SIZE_TEST = image_size

        if image_size <= 384:
            print('\n' + '='*80)
            print(f'[INFO] Anchor size reduced to  [[8, 16, 32, 64, 128]] for input size {image_size}')
            print('='*80)
            self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]

    @staticmethod
    def _register_datasets(data_dir: str, split: str) -> str:
        name = f'{os.path.basename(data_dir)}_{split}'
        json_file = os.path.join(data_dir, split, '_annotations.coco.json')
        image_root = os.path.join(data_dir, split)

        if name not in DatasetCatalog.list():
            if os.path.exists(json_file):
                register_coco_instances(name, {}, json_file, image_root)
            else:
                raise FileNotFoundError(f'  - ERROR: {json_file} not found.')

        return name


class DetectronR50Predictor(DetectronPredictor):
    """Predictor using Detectron2's Mask R-CNN with ResNet-50 backbone for instance segmentation."""
    def __init__(self, model_path: str | None = None):
        super().__init__('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', model_path)

    def get_name(self) -> str:
        return super().get_name() + '-resnet-50'


class DetectronR101Predictor(DetectronPredictor):
    """Predictor using Detectron2's Mask R-CNN with ResNet-101 backbone for instance segmentation."""
    def __init__(self, model_path: str | None = None):
        super().__init__('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml', model_path)

    def get_name(self) -> str:
        return super().get_name() + '-resnet-101'


class DetectronResnextPredictor(DetectronPredictor):
    """Predictor using Detectron2's Mask R-CNN with ResNeXt-101 backbone for instance segmentation."""
    def __init__(self, model_path: str | None = None):
        super().__init__('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml', model_path)

    def get_name(self) -> str:
        return super().get_name() + '-resnext'
