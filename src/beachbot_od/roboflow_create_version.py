import roboflow
import keyring
import os
import yaml

# See https://docs.roboflow.com/api-reference/versions/create-a-project-version for more information


def load_config(config_path):
    # Load the YAML configuration file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def generate_version(config_path="../roboflow_version_config.yaml"):
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

    project = rf.workspace().project("beach-cleaning-object-detection")

    version: int
    # Load default config settings
    settings = load_config(config_path=config_path)

    version = project.generate_version(settings)
    print("Version ID:", version)


if __name__ == "__main__":
    generate_version()
