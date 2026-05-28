import argparse
import os

from predictors import PREDICTORS, create_predictor, Predictor


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train leaf-segmentation models using YOLO26, Detectron2 or RF-Detr.')

    parser.add_argument('--architecture', type=str, choices=PREDICTORS, required=True,
                        help='Model architecture to be used for training.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory containing images, annotations and split files in yolo or coco format, '
                             'depending on selected architecture.')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory for saving training results in subdirectories '
                             'according to architecture and image size.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--input_sizes', nargs='+', type=int, default=[192, 384, 576, 768],
                        help='Image sizes for training, resulting in a separate model for each size.')

    return parser.parse_args()


def train(predictor: Predictor, data_dir: str, output_dir: str, epochs: int, batch_size: int,
          input_sizes: list[int]):
    """Trains the specified predictor architecture on the provided dataset for multiple input sizes.
    Args:
        predictor (Predictor): The predictor instance to be trained.
        data_dir (str): Path to the training dataset in format corresponding to the respective architecture.
        output_dir (str): Directory where the trained model and logs will be saved.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per training batch.
        input_sizes (list[int]): Image sizes to be used for training, resulting in a separate model for each size.
    """
    for imgsz in input_sizes:
        print('\n' + '='*80)
        print(f'[INFO] Start Training: {predictor.get_name()} @ {imgsz}px')
        print('='*80)

        try:
            predictor.train(
                data_path=data_dir,
                image_size=imgsz,
                output_path=os.path.join(output_dir, predictor.get_name(), f'imgsz-{imgsz}'),
                batch_size=batch_size,
                epochs=epochs
            )
            print(f'[OK] Training for {predictor.get_name()} with image size {imgsz} completed successfully.')

        except KeyboardInterrupt:
            print('  - ERROR: Training interrupted by user. Exiting.')
            return


if __name__ == "__main__":
    args = parse_arguments()
    pred = create_predictor(args.architecture)
    train(pred, args.data_dir, args.output_dir, args.epochs, args.batch, args.input_sizes)
