from huggingface_hub import hf_hub_download
from pathlib import Path


def get_model(model_name="yolo_v5_beachbot_160") -> Path:
    repo_path = f"okinawa-ai-beach-robot/{model_name}"
    model_path = hf_hub_download(repo_id=repo_path, filename="model.pt")
    model_path = Path(model_path)
    return model_path


if __name__ == "__main__":
    get_model(model_name="yolo_v5_beachbot_160")
