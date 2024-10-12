import roboflow
import keyring
import os


def get_dataset(ver=1, model_format="coco", location="./dataset", overwrite=False):
    """
    Downloads dataset from Roboflow
    """

    # Access API from environment variable
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

    project = rf.workspace("okinawaaibeachrobot").project(
        "beach-cleaning-object-detection"
    )
    version = project.version(ver)
    # Show warning if not overwriting and location already exists so users know new data will not be used
    if not overwrite and os.path.exists(location):
        print(
            """
        WARNING: dataset directory already exists, not overwriting.
        To overwrite, set overwrite=True
        """
        )

    dataset = version.download(model_format, location, overwrite)
    breakpoint()
    print("Dataset downloaded at " + dataset.location)


if __name__ == "__main__":
    get_dataset()
