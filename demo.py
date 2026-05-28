import os
import argparse
from tqdm import tqdm
import cv2

from predictors import PREDICTORS, create_predictor, Predictor

from utils.visualization import visualize


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Apply a leaf-segmentation model on a directory of images and visualize results.')

    # Format and Framework
    parser.add_argument('--architecture', type=str, required=True, choices=PREDICTORS,
                        help='Architecture of trained model.')

    # Paths
    parser.add_argument('--model', type=str, required=True, help='Path to trained model file.')
    parser.add_argument('--images_dir', required=True, type=str, help='Path to input images directory.')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Directory to save results.')

    # Modes and Parameters
    parser.add_argument('--image_size', type=int, default=None,
                        help='Image size for inference. If None, architecture default will be used.')
    parser.add_argument('--confidence', type=float, default=None,
                        help='Confidence threshold for inference. '
                             'If not set, the default threshold for the architecture will be used.')
    parser.add_argument('--show_boxes', type=bool, default=False,
                        help='Whether to display bounding boxes in the output images.')

    return parser.parse_args()


def process(predictor: Predictor, images_dir: str, output_path: str, image_size: int, confidence: float | None = None,
            show_boxes: bool = False):
    """Processes the input images using the specified predictor, visualizes the predictions, and saves the results.
    Args:
        predictor (Predictor): The predictor instance to use for inference.
        images_dir (str): Path to the directory containing input images.
        output_path (str): Path to the directory where output images and annotations will be saved.
        image_size (int): Image size to be used for inference. If None, architecture default will be used.
        confidence (float | None): Confidence threshold for filtering predictions. 
                                   If None, the default threshold of the predictor will be used.
        show_boxes (bool): Whether to draw bounding boxes around predicted objects in the output images.
    """
    os.makedirs(output_path, exist_ok=True)

    image_names = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))]

    for img_name in tqdm(image_names, desc='processing images', unit='img'):
        image_path = os.path.join(images_dir, img_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f'  - WARNING: Could not read image "{image_path}"')
            continue

        results = predictor.predict(image, image_size, confidence)[0]
        yolo_output = visualize(image, results, show_boxes)
        cv2.imwrite(os.path.join(output_path, img_name), image)
        with open(os.path.join(output_path, f'{os.path.splitext(img_name)[0]}.txt'), 'w') as output_file:
            output_file.write('\n'.join(yolo_output))

    print(f"\n[OK] Processing finished. All files saved to: {output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    process(create_predictor(args.architecture, args.model), args.images_dir,
            args.output_dir, args.image_size, args.confidence, args.show_boxes)
