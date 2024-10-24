from beachbot_od.utils.models import (
    get_model_path,
    model_exists,
    get_base_model_weights_path,
)
from yolov5 import train
import shutil
from pathlib import Path
import keyring


def run(
    model_format: str = "yolov5s",
    img_width: int = 160,
    dataset_version: int = 13,
    epochs: int = 1,
    overwrite: bool = False,
):
    """
    Train model using yolov5-pip's train.run
    Saves models to standardized BEACHBOT_MODELS directory
    and removes previous train outputs if overwrite=True

    Pulls datasets from roboflow if not available locally
    Stores them in the pip installation for yolov5-pip
    e.g. yolov5-pip/yolov5/beach-cleaning-object-detection-13

    Args:
        model_format: str
        imgsz: int
        dataset_version: int
        epochs: int
        overwrite: bool
    Returns:

    """

    # Just a tmp dir to use for the rigid save_dir definition in train.run
    runs_dir: str = "beachbot_train_runs"

    # Get Base Model
    base_model_weights_path = get_base_model_weights_path(
        model_format=model_format,
        overwrite=overwrite,
    )

    # Get standard model path from BEACHBOT_MODELS
    model_path = get_model_path(
        model_format=model_format,
        dataset_version=dataset_version,
        img_width=img_width,
    )

    # Checks if model already exists (first locally then on huggingface)
    if model_exists(model_path) and not overwrite:
        raise Exception(
            f"""
        Model {model_path} already exists. Aborting.
        Add overwrite=True arg to overwrite."
        """
        )

    # Remove previous train outputs from runs_dir if overwrite=True
    # else intentionally fail to prevent indexing and losing track
    # of which training pertains to which in the standard yolov5
    # output scheme.
    if Path(runs_dir).exists():
        if overwrite:
            shutil.rmtree(Path(runs_dir))
        else:
            raise Exception(
                f"""The directory '{runs_dir}' already exists.
            Aborting to avoid overwriting.
            Add overwrite=True arg to overwrite.
            """
            )

    # Tested and this uses pretrained weights for yolov5s even without,
    # specifying weights. See details on where these are from here:
    # https://github.com/fcakyon/yolov5-pip/blob/7663793e49a9392dfe951c9ca8f631b01d7793ae/yolov5/utils/downloads.py#L84
    # Though I chose to use a copy of the base model within the BEACHBOT_MODELS
    # directory as otherwise a local copy is downloaded to the current working
    # directory which is not ideal in many situations.

    opt = train.run(
        weights=base_model_weights_path,
        imgsz=img_width,
        data=f"https://universe.roboflow.com/okinawaaibeachrobot/beach-cleaning-object-detection/dataset/{dataset_version}",
        roboflow_token=keyring.get_password("roboflow", "api_key"),
        epochs=epochs,
        exist_ok=True,
        project=runs_dir,
    )

    # Move results to BEACHBOT_MODELS
    print(f"Copying {opt.save_dir} to {model_path}")
    shutil.copytree(Path(opt.save_dir), Path(model_path), dirs_exist_ok=True)
    shutil.rmtree(Path(runs_dir))


if __name__ == "__main__":
    run()
