from platformdirs import PlatformDirs
import os


# Define the directory paths you want to make available globally
_platform_dirs = PlatformDirs("beachbot_od", "okinawa-ai-beach-robot")
HOME = _platform_dirs.user_data_dir
CACHE_DIR = _platform_dirs.user_cache_dir
CONFIG_DIR = _platform_dirs.user_config_dir
LOGS_DIR = _platform_dirs.user_log_dir
MODELS_DIR = CACHE_DIR + "/models"
DATASET_DIR = CACHE_DIR + "/datasets"

# Ensure the directories exist
os.makedirs(HOME, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

# Optionally print for debugging (remove in production)
print(f"HOME: {HOME}")
print(f"CACHE_DIR: {CACHE_DIR}")
print(f"CONFIG_DIR: {CONFIG_DIR}")
print(f"LOGS_DIR: {LOGS_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"DATASET_DIR: {DATASET_DIR}")
