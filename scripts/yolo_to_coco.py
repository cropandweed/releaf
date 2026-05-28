import json
import os
import shutil
import argparse
import yaml
from tqdm import tqdm
from PIL import Image
from datetime import datetime

CODE_VERSION = '1.5'
DATA_SPLITS = ['train', 'val', 'test']


def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert YOLO Segmentation annotations to COCO format.')

    parser.add_argument('--data_dir', required=True, type=str,
                        help='Path to directory containing images, annotations and '
                             'split files (test.txt, train.txt or val.txt) in YOLO format.')
    parser.add_argument('--dataset_config', default=None, type=str,
                        help='Path to YOLO dataset configuration in yaml format containing category names.'
                             'Defaults to dataset_config.yaml in the data_dir if not provided.')
    parser.add_argument('--output_dir', required=False, type=str, default='./data',
                        help='Directory where the results will be saved.')
    parser.add_argument('--merge_categories', action='store_true',
                        help='Whether to merge all categories into a single category in the output COCO dataset.')

    input_args = parser.parse_args()
    if input_args.dataset_config is None:
        input_args.dataset_config = os.path.join(input_args.data_dir, 'dataset_config.yaml')
    return input_args


def read_categories(dataset_config: str) -> list[dict]:
    """Reads the category names from a YOLO labels file and converts them to COCO category format.
    Args:
        dataset_config (str): Path to YOLO dataset configuration file in yaml format.
    Returns:
        list[dict]: List of COCO category dictionaries.
    """

    with open(dataset_config, 'r') as f:
        data = yaml.safe_load(f)

    names = data.get('names', {})
    if len(names) == 0:
        print(f'Warning: No category names found in file {dataset_config}.')

    coco_categories = [{
        "id": int(idx) + 1,  # COCO label ids are starting with 1
        "name": name,
        "subcategory": ""
    } for idx, name in names.items()]

    return coco_categories


def prepare_coco_annotation(x_coords: list[float], y_coords: list[float],
                            width: int, height: int) -> tuple[list[int], float, list[list[int]]]:
    """Converts normalized YOLO polygon coordinates to absolute COCO geometry data.
       Calculates the bounding box, the exact polygon area, and the segmentation array.
    Args:
        x_coords (list[float]): List of normalized x coordinates of the polygon vertices.
        y_coords (list[float]): List of normalized y coordinates of the polygon vertices.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
    Returns:
        tuple: (bbox, area, segmentation)
            - bbox (list[int]): [x_min, y_min, width, height] in pixels.
            - area (float): Exact area of the polygon (shoelace formula).
            - segmentation (list[list[int]]): Nested list of [x, y, x, y, ...] coordinates.
    """
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    bbox = [
        int(x_min * width),
        int(y_min * height),
        int((x_max - x_min) * width),
        int((y_max - y_min) * height)
    ]

    area = bbox[2] * bbox[3]
    segmentation = []

    for x, y in zip(x_coords, y_coords):
        segmentation.append(x * width)
        segmentation.append(y * height)

    return bbox, area, [segmentation]


SPLIT_NAMES = {
    'val': 'valid',
}


def generate_coco_dataset(data_dir: str, dataset_config: str, output_dir: str, split_name: str,
                          merge_categories: bool = False):
    """Converts a specific YOLO data split to the COCO format,
       Copies the images, and creates the associated annotation JSON.   
    Args:
        data_dir (str): Path to the dataset directory containing the split text files and images.
        dataset_config (str): Path to the YOLO dataset configuration file in yaml format.
        output_dir (str): Directory for saving results.
        split_name (str): Name of the data split (e.g., 'train', 'val', 'test').
        merge_categories (bool): Whether to merge all categories into a single category in the output COCO dataset.
    """

    split_output_dir = os.path.join(output_dir, SPLIT_NAMES.get(split_name, split_name))
    os.makedirs(split_output_dir, exist_ok=True)

    coco_dataset = {
        "info": {
            "description": "YOLO to COCO Conversion",
            "version": CODE_VERSION,
            "year": datetime.now().year,
            "date_created": datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        },
        "licenses": [],
        "categories": [{
            "id": 1,
            "name": "vegetation",
            "subcategory": ""
        }] if merge_categories else read_categories(dataset_config),
        "images": [],
        "annotations": []
    }

    image_list = os.path.join(data_dir, f'{split_name}.txt')
    if not os.path.exists(image_list):
        return

    with open(image_list, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]

    annotation_id = 1
    image_id = 1

    for img_rel_path in tqdm(image_paths, desc=f"Processing {split_name} images", unit='img'):
        img_full_path = os.path.join(data_dir, img_rel_path)

        if not os.path.exists(img_full_path):
            continue

        image = Image.open(img_full_path)
        width, height = image.size
        filename = os.path.basename(img_rel_path)

        shutil.copy(img_full_path, os.path.join(split_output_dir, filename))

        coco_dataset["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # Label path logic (YOLO to COCO)
        label_dir = os.path.dirname(img_full_path).replace('images', 'labels')
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                annotations = f.readlines()

            for annotation in annotations:
                parts = annotation.strip().split()
                if len(parts) < 5:
                    continue

                class_id = 1 if merge_categories else (int(parts[0]) + 1)
                norm_coords = list(map(float, parts[1:]))
                bbox, area, [seg] = prepare_coco_annotation(norm_coords[0::2], norm_coords[1::2], width, height)

                coco_dataset["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": [seg]
                })
                annotation_id += 1

        image_id += 1

    with open(os.path.join(split_output_dir, '_annotations.coco.json'), 'w') as f:
        json.dump(coco_dataset, f, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    for split in DATA_SPLITS:
        generate_coco_dataset(args.data_dir, args.dataset_config, args.output_dir, split, args.merge_categories)
