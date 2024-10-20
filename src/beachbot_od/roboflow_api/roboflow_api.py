import roboflow
import keyring
import os
import yaml

from ..config import BEACHBOT_DATASETS


def load_config(config_path):
    # Load the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def connect() -> roboflow.Roboflow:
    # Access API from huggingface secrets
    api_key = os.getenv("ROBOFLOW_KEY")

    if not api_key:
        # For local installations use keyring to securely store the API key
        api_key = keyring.get_password("roboflow", "api_key")

    if not api_key:
        raise ValueError(
            """API key not found in keyring. 
        Make sure it is stored securely.
        HuggingFace Usage:
        os.getenv("ROBOFLOW_API_KEY")
        Local Usage: 
        pip install keyring
        python -c "import keyring; keyring.set_password('roboflow', 'api_key', 'your_actual_api_key')"
        """
        )
    rf = roboflow.Roboflow(api_key)
    return rf


def generate_version(config_path="../roboflow_version_config.yaml") -> int:
    # Return version ID
    # See https://docs.roboflow.com/api-reference/versions/create-a-project-version for more information

    rf = connect()
    project = rf.workspace().project("beach-cleaning-object-detection")

    version: int
    # Load default config settings
    settings = load_config(config_path=config_path)

    version = project.generate_version(settings)
    print("Version ID:", version)
    return version


def get_dataset(ver: int = 1, model_format="coco", location=None, overwrite=False):
    """
    Downloads dataset from Roboflow
    """
    rf = connect()
    project = rf.workspace("okinawaaibeachrobot").project(
        "beach-cleaning-object-detection"
    )
    version = project.version(ver)

    # Show warning if not overwriting and location already exists so users know new data will not be used
    if location is None:
        location = BEACHBOT_DATASETS + "/" + str(ver) + "/" + model_format
        print(f"Dataset location not specified, using default location: {location}.")
    if os.path.exists(location) and not overwrite:
        print(
            f"""
        WARNING: dataset directory already exists at {location}.
        Will not overwrite. To overwrite, set overwrite=True
        """
        )
    else:
        print("Dataset will be downloaded to " + location)

    dataset = version.download(model_format, location, overwrite)
    print("Dataset located at " + dataset.location)
