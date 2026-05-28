import os
import argparse
import json
from tqdm import tqdm
import cv2
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from predictors import PREDICTORS, create_predictor, Predictor
from utils.metrics import binary_mask_to_rle, symmetric_best_dice, MaskStats, BoxStats


def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate a leaf-segmentation model based on multiple metrics.')

    parser.add_argument('--architecture', choices=PREDICTORS, required=True, help='Architecture of trained model.')
    parser.add_argument('--model', required=True, help='Path to trained model file.')
    parser.add_argument('--output_dir', default='./results',
                        help='Directory for saving results in subdirectories according to architecture and image size.')
    parser.add_argument('--gt_path', required=True,
                        help='Path to ground-truth JSON file in COCO format and images in the same directory.')
    parser.add_argument('--image_size', type=int, default=576, help='Image size for inference.')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Confidence threshold for predictions. '
                             'If not set, the default threshold for the architecture will be used.')
    return parser.parse_args()


def val(predictor: Predictor, gt: COCO, images_dir: str, output_dir: str, image_size: int,
        confidence: float | None = None):
    """Evaluates the predictor on the given COCO dataset by running inference on each image, 
       comparing predictions to ground truth annotations, and calculating evaluation metrics.
    Args:
        predictor (Predictor): The predictor instance to be evaluated.
        gt (COCO): The COCO object containing ground-truth annotations for the dataset.
        images_dir (str): Path to the directory containing input images corresponding to annotations.
        output_dir (str): Path to the directory where evaluation results will be saved.
        image_size (int): Image size to be used for inference. If None, architecture default will be used.
        confidence (float | None): Confidence threshold for filtering predictions. 
                                   If None, the default threshold of the predictor will be used.
    """
    print('[INFO] benchmarking latency...')
    avg_ms = predictor.measure_latency(image_size, confidence=confidence)
    print(f'[INFO] Running inference on {len(gt.getImgIds())} images...')

    results = []
    all_sbd = []
    m_stats = MaskStats()
    b_stats = BoxStats()

    coco_results = []

    for img_info in tqdm(gt.loadImgs(gt.getImgIds()), desc='processing images', unit='img'):
        image = cv2.imread(os.path.join(images_dir, img_info['file_name']))
        img_id = img_info['id']

        anno = gt.loadAnns(gt.getAnnIds(imgIds=[img_id], catIds=[1]))
        gt_masks = [gt.annToMask(a).astype(bool) for a in anno]

        boxes = [a['bbox'] for a in anno]
        gt_boxes = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]

        results = predictor.predict(image, image_size, confidence)[0]

        p_boxes = []
        p_masks = []
        for result in results:
            if result.mask.shape[:2] != image.shape[:2]:
                result.mask = cv2.resize(result.mask.astype(np.uint8), (image.shape[1], image.shape[0]),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)

            box = result.get_box()
            x1, y1, x2, y2 = box
            coco_results.append({
                'image_id': img_id,
                'category_id': 1,
                'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                'score': result.score,
                'segmentation': binary_mask_to_rle(result.mask)
            })
            p_boxes.append(box)
            p_masks.append(result.mask)

        all_sbd.append(symmetric_best_dice(gt_masks, p_masks))
        m_stats.add(gt_masks, p_masks)
        b_stats.add(gt_boxes, p_boxes)

    results_path = os.path.join(output_dir, predictor.get_name(), f'imgsz-{image_size}')
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    res_path = os.path.join(results_path, 'results.json')
    with open(res_path, 'w') as f:
        json.dump(coco_results, f)
    coco_dt = gt.loadRes(res_path)

    print(f'\n{'='*20} RESULTS: {predictor.get_name()} {'='*20}')
    ev_b = COCOeval(gt, coco_dt, 'bbox')
    ev_b.evaluate()
    ev_b.accumulate()
    ev_b.summarize()

    ev_m = COCOeval(gt, coco_dt, 'segm')
    ev_m.evaluate()
    ev_m.accumulate()
    ev_m.summarize()

    print(f'\n[INFO] Instance SBD: {np.mean(all_sbd):.4f}')
    print('[INFO] Mask Stats', m_stats)

    print('[INFO] Box Stats ', b_stats)
    print(f'[INFO] Inference Time: {avg_ms:.2f} ms/img')
    print(f'[INFO] FPS:            {1000/avg_ms:.1f}')
    print(f'{'='*50}\n')


if __name__ == "__main__":
    args = parse_arguments()
    pred = create_predictor(args.architecture, args.model)
    gt_coco = COCO(args.gt_path)
    val(pred, gt_coco, os.path.split(args.gt_path)[0], args.output_dir, args.image_size, args.confidence)
