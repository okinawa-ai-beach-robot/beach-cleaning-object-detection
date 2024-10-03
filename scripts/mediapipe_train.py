import os
import json
import shutil
import requests
import zipfile
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
from collections import defaultdict
import math
import tensorflow as tf
from mediapipe_model_maker import object_detector as mm_object_detector
from mediapipe.tasks.python import vision
import mediapipe as mp
import cv2
import argparse

# Constants
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red
OUTPUT_DIR = "./"
TRAIN_DATASET_PATH = "data/train"
VALIDATION_DATASET_PATH = "data/valid"
TEST_DATASET_PATH = "data/test"
DEFAULT_EPOCHS = 1
DEFAULT_DATASET_URL_KEY = "ALL_CLASSES"

# Define dataset URLs
DATASET_URLS = {
    "ALL_CLASSES": "https://app.roboflow.com/ds/CFJu2uGHhw?key=H6r8QeWRsy",
    "NO_BOUNDARY_CLASSES": "https://app.roboflow.com/ds/drjuueUVSw?key=30gLPNigfm",
    "HARD_HAT_TEST": "https://app.roboflow.com/ds/2BJ5i4Tmxc?key=y58GLhUEln",
}

assert tf.__version__.startswith("2")

def parse_args():
    parser = argparse.ArgumentParser(description="Object Detection Training Script")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs for training")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_URL_KEY, choices=DATASET_URLS.keys(), help="Dataset URL key to use")
    return parser.parse_args()

def init(dataset_url: str) -> None:
    """Initial setup: remove existing data, download, and extract dataset."""
    shutil.rmtree("data", ignore_errors=True)
    zip_path = "roboflow.zip"

    # Download dataset
    response = requests.get(dataset_url, stream=True)
    with open(zip_path, "wb") as file:
        file.write(response.content)

    # Unzip and clean up
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall("data")
    os.remove(zip_path)

    # Process files
    for directory in ["test", "train", "valid"]:
        process_directory(directory)

    # Remove class id=0 for MediaPipe compatibility
    for file in glob.glob("data/*/labels.json"):
        remove_category(file)

    os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_directory(directory: str) -> None:
    """Process image files and annotations in a directory."""
    src_dir = f"data/{directory}"
    dest_dir = os.path.join(src_dir, "images")
    os.makedirs(dest_dir, exist_ok=True)

    for ext in ["jpg", "jpeg", "png"]:
        for file_path in glob.glob(os.path.join(src_dir, f"*.{ext}")):
            shutil.move(file_path, dest_dir)

    shutil.move(
        os.path.join(src_dir, "_annotations.coco.json"),
        os.path.join(src_dir, "labels.json"),
    )


def remove_category(file_path: str, category_id_to_remove: int = 0) -> None:
    """Remove the specific category from a JSON file."""
    with open(file_path, "r") as file:
        data = json.load(file)

    if data["categories"][0]["id"] == category_id_to_remove:
        del data["categories"][0]

    with open(file_path, "w") as file:
        json.dump(data, file)


def draw_bboxes(
    ax, annotations, cat_id_to_label, image_shape, color="red", is_ground_truth=True
) -> None:
    """Draw bounding boxes and labels on the image."""
    for annotation in annotations:
        if is_ground_truth:
            # Ground truth format
            bbox = annotation["bbox"]
            x, y, w, h = bbox
            label = cat_id_to_label.get(annotation.get("category_id", -1), "unknown")
        else:
            # Inference result format (Detection object)
            bbox = annotation.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            label = annotation.categories[0].category_name

        # Draw the bounding box
        rect = plt.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Draw label text
        ax.text(
            x,
            y - 10,
            label,
            color=color,
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor=color),
        )


def plot_ground_truth_and_inferences(
    dataset_folder: str, model: mm_object_detector.ObjectDetector, output_path: str
) -> None:
    """Plot the first 9 images with both ground truth and inferred bounding boxes."""
    with open(os.path.join(dataset_folder, "labels.json"), "r") as file:
        labels_json = json.load(file)

    images = labels_json["images"]
    cat_id_to_label = {item["id"]: item["name"] for item in labels_json["categories"]}
    image_annots = defaultdict(list)

    for annotation in labels_json["annotations"]:
        image_annots[annotation["image_id"]].append(annotation)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i, (image_id, annotations_list) in enumerate(list(image_annots.items())[:9]):
        ax = axs[i // 3, i % 3]
        img_path = os.path.join(dataset_folder, "images", images[image_id]["file_name"])
        image = plt.imread(img_path)
        ax.imshow(image)

        # Draw ground truth bounding boxes (red)
        draw_bboxes(
            ax,
            annotations_list,
            cat_id_to_label,
            image.shape,
            color="red",
            is_ground_truth=True,
        )

        # Perform inference and draw inferred bounding boxes (green)
        detected_results = model.detect(mp.Image.create_from_file(img_path))
        draw_bboxes(
            ax,
            detected_results.detections,
            cat_id_to_label,
            image.shape,
            color="green",
            is_ground_truth=False,
        )

        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def evaluate(model, validation_data, single_class=False) -> None:
    """Evaluate the model and save results."""
    if single_class:
        filter_trash_easy_annotations(os.path.join(VALIDATION_DATASET_PATH, "labels.json"))
        validation_data = mm_object_detector.Dataset.from_coco_folder(
            VALIDATION_DATASET_PATH, cache_dir="/tmp/od_data/validation"
        )
        print("filtered_validation_data size: ", validation_data.size)
        output_file_path = os.path.join(OUTPUT_DIR, "filtered_results.txt")
    else:
        output_file_path = os.path.join(OUTPUT_DIR, "results.txt")

    loss, coco_metrics = model.evaluate(validation_data, batch_size=4)

    with open(output_file_path, "w") as file:
        file.write(f"Validation loss: {loss}\n")
        file.write(f"Validation coco metrics: {coco_metrics}\n")

    print(f"Validation loss: {loss}")
    print(f"Validation coco metrics: {coco_metrics}")
    print(f"Results saved to {output_file_path}")


def show_classes() -> None:
    """Print class labels from the training dataset."""
    with open(os.path.join(TRAIN_DATASET_PATH, "labels.json"), "r") as file:
        labels_json = json.load(file)
        print("Classes: -------------------")
    for category in labels_json["categories"]:
        print(f"{category['id']}: {category['name']}")


def filter_trash_easy_annotations(
    input_json_path: str, category_name: str = "trash_easy"
) -> None:
    """Filter annotations to keep only those matching the specified category."""
    with open(input_json_path, "r") as file:
        coco_data = json.load(file)

    category_id = next(
        (cat["id"] for cat in coco_data["categories"] if cat["name"] == category_name),
        None,
    )
    if category_id is None:
        print(f"Category '{category_name}' not found in the JSON file.")
        return

    filtered_annotations = [
        ann for ann in coco_data["annotations"] if ann["category_id"] == category_id
    ]
    filtered_coco_data = {
        "images": coco_data["images"],
        "annotations": filtered_annotations,
        "categories": [
            cat for cat in coco_data["categories"] if cat["id"] == category_id
        ],
    }

    with open(input_json_path, "w") as file:
        json.dump(filtered_coco_data, file, indent=4)

    print(f"Filtered data saved to {input_json_path}")


def create_model(epochs: int) -> mm_object_detector.ObjectDetector:
    """Create and return an object detection model."""
    spec = mm_object_detector.SupportedModels.MOBILENET_MULTI_AVG
    # Saving best checkpoint: See https://github.com/google-ai-edge/mediapipe/issues/4912
    # best_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    #    filepath=os.path.join(output_dir + "/checkpoint_best", "keras_model"),
    #    monitor="total_loss",
    #    save_best_only=True,
    #    verbose=1,
    # )
    train_data = mm_object_detector.Dataset.from_coco_folder(
        TRAIN_DATASET_PATH, cache_dir="/tmp/od_data/train"
    )
    validation_data = mm_object_detector.Dataset.from_coco_folder(
        VALIDATION_DATASET_PATH, cache_dir="/tmp/od_data/validation"
    )
    print("train_data size: ", train_data.size)
    print("validation_data size: ", validation_data.size)
    hparams = mm_object_detector.HParams(
        export_dir=OUTPUT_DIR + "checkpoints", epochs=epochs
    )
    options = mm_object_detector.ObjectDetectorOptions(
        supported_model=spec, hparams=hparams
    )
    model = mm_object_detector.ObjectDetector.create(
        train_data=train_data, validation_data=validation_data, options=options
    )
    model_path = os.path.join(OUTPUT_DIR, "model.tflite")
    model.export_model(model_path)
    return model


def create_detector() -> vision.ObjectDetector:
    """Create and return a MediaPipe object detector."""
    BaseOptions = mp.tasks.BaseOptions
    ObjectDetector = mp.tasks.vision.ObjectDetector
    ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    model_path = os.path.join(OUTPUT_DIR, "checkpoints/model.tflite")
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        max_results=5,
        running_mode=VisionRunningMode.IMAGE,
    )
    return ObjectDetector.create_from_options(options)


if __name__ == "__main__":
    args = parse_args()
    init(DATASET_URLS[args.dataset])
    show_classes()
    model = create_model(args.epochs)
    evaluate(
        model, mm_object_detector.Dataset.from_coco_folder(VALIDATION_DATASET_PATH)
    )
    evaluate(
        model, mm_object_detector.Dataset.from_coco_folder(VALIDATION_DATASET_PATH), single_class=True
    )
    detector = create_detector()
    plot_ground_truth_and_inferences(TEST_DATASET_PATH, detector, "detector_sample.png")
