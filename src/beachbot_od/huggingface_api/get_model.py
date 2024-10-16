from huggingface_hub import hf_hub_download
from pathlib import Path


def get_model(version: str, resolution: int) -> Path:
    model_name = f"yolo_v5_beachbot_{resolution}"
    repo_path = f"okinawa-ai-beach-robot/{model_name}"
    model_path = hf_hub_download(
        repo_id=repo_path, filename="model.pt", revision=version
    )
    model_path = Path(model_path)
    return model_path


if __name__ == "__main__":
    get_model(version="v1.0", resolution=160)
