from beachbot_od.roboflow_api import get_dataset
from beachbot_od.utils.models import get_model_path, get_base_model_weights_path
from roboflow.core.dataset import Dataset
from yolov5 import train
import shutil
from pathlib import Path


def run(
    model_format: str = "yolov5s",
    img_width: int = 160,
    dataset_version: int = 13,
    dataset_format: str = "yolov5pytorch",
    epochs: int = 1,
    overwrite: bool = False,
):
    """
    Train model

    Args:
        model_format: str
        imgsz: int
        dataset_version: int
        dataset_format: str
        epochs: int
        overwrite: bool
    Returns:

    """

    ###
    # Just a tmp directory to use for the rigid save_dir definition in
    # yolov5.train.run
    # The only reason I overwrite the default is to be able to remove
    # it prior to running train
    ###
    runs_dir: str = "beachbot_train_runs"

    # Get Base Model
    base_model_weights_path = get_base_model_weights_path(
        model_format=model_format,
        overwrite=overwrite,
    )

    # Get Model (leave here before training as it will throw an error on unintentional overwriting)
    model_path = get_model_path(
        model_format=model_format,
        dataset_version=dataset_version,
        imgsz=img_width,
        overwrite=overwrite,
    )

    # Load dataset
    dataset: Dataset = get_dataset(ver=dataset_version, dataset_format=dataset_format)
    dataset_yaml = dataset.location + "/data.yaml"

    # Remove previous train outputs
    if Path(runs_dir).exists():
        shutil.rmtree(Path(runs_dir))

    # Tested and this uses pretrained weights, see details on where these are from here:
    # https://github.com/fcakyon/yolov5-pip/blob/7663793e49a9392dfe951c9ca8f631b01d7793ae/yolov5/utils/downloads.py#L84
    opt = train.run(
        weights=base_model_weights_path,
        imgsz=img_width,
        data=dataset_yaml,
        epochs=epochs,
        exist_ok=True,
        project=runs_dir,
        rect=True,
    )

    # Move results to BEACHBOT_MODELS
    print(f"Copying {opt.save_dir} to {model_path}")
    shutil.copytree(Path(opt.save_dir), Path(model_path), dirs_exist_ok=True)
    shutil.rmtree(Path(runs_dir))


if __name__ == "__main__":
    run()
