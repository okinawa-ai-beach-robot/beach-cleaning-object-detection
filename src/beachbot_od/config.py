from platformdirs import PlatformDirs
import os


# Define the directory paths you want to make available globally
_platform_dirs = PlatformDirs("beachbot_od", "okinawa-ai-beach-robot")
if os.getenv("BEACHBOT_HOME"):
    BEACHBOT_HOME = os.getenv("BEACHBOT_HOME")
else:
    BEACHBOT_HOME = _platform_dirs.user_data_dir

if os.getenv("BEACHBOT_CACHE"):
    BEACHBOT_CACHE = os.getenv("BEACHBOT_CACHE")
else:
    BEACHBOT_CACHE = _platform_dirs.user_cache_dir

if os.getenv("BEACHBOT_CONFIG"):
    BEACHBOT_CONFIG = os.getenv("BEACHBOT_CONFIG")
else:
    BEACHBOT_CONFIG = _platform_dirs.user_config_dir

if os.getenv("BEACHBOT_LOGS"):
    BEACHBOT_LOGS = os.getenv("BEACHBOT_LOGS")
else:
    BEACHBOT_LOGS = _platform_dirs.user_log_dir

if os.getenv("BEACHBOT_MODELS"):
    BEACHBOT_MODELS = os.getenv("BEACHBOT_MODELS")
else:
    BEACHBOT_MODELS = BEACHBOT_CACHE + "/models"

if os.getenv("BEACHBOT_DATASETS"):
    BEACHBOT_DATASETS = os.getenv("BEACHBOT_DATASETS")
else:
    BEACHBOT_DATASETS = BEACHBOT_CACHE + "/datasets"

# Ensure the directories exist
os.makedirs(BEACHBOT_HOME, exist_ok=True)
os.makedirs(BEACHBOT_CACHE, exist_ok=True)
os.makedirs(BEACHBOT_CONFIG, exist_ok=True)
os.makedirs(BEACHBOT_LOGS, exist_ok=True)
os.makedirs(BEACHBOT_MODELS, exist_ok=True)
os.makedirs(BEACHBOT_DATASETS, exist_ok=True)

# Optionally print for debugging (remove in production)
print(f"BEACHBOT_HOME: {BEACHBOT_HOME}")
print(f"BEACHBOT_CACHE: {BEACHBOT_CACHE}")
print(f"BEACHBOT_CONFIG: {BEACHBOT_CONFIG}")
print(f"BEACHBOT_LOGS: {BEACHBOT_LOGS}")
print(f"BEACHBOT_MODELS: {BEACHBOT_MODELS}")
print(f"BEACHBOT_DATASETS: {BEACHBOT_DATASETS}")
