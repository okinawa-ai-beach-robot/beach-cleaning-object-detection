from beachbot_od.roboflow_api import connect, load_config, generate_version, get_dataset

print("Connecting to Roboflow...")
rf = connect()
if rf is None:
    print("Failed to connect to Roboflow")
else:
    print("Connected to Roboflow")

# Ensure you are running the test from the test dir
print("Loading config...")
config = load_config("../config/roboflow_version_config.yaml")
print("Loaded ../config/roboflow_version_config.yaml:")
print(config)

# Need a way to test `generate_version` without actually generating a new one. Guess there is some test arg for this
# rf.generate_version(config_path="roboflow_version_config_test.yaml")

# 13 being the dummy tiny dataset
get_dataset(ver=13)
