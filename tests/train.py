from beachbot_od.yolo_v5_api.train import run

run(model_format="yolov5s", img_width=160, dataset_version=13, epochs=1, overwrite=True)
