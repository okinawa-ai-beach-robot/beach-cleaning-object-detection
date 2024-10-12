from globox import AnnotationSet, COCOEvaluator, BoxFormat
import sys
import torch


def model_evaluate(model_path, gt_label_path):
    gt = AnnotationSet.from_coco(
        file_path=gt_label_path,
    )
    images = ["dataset/test/" + item for item in list(gt.image_ids)]

    # Load model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

    # set model parameters
    model.conf = 0.70  # NMS confidence threshold
    model.iou = 0.50  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = 10  # maximum number of detections per image

    ## Run inference on all images within dataset
    results = model(images)
    results.save(save_dir="./detection_images", exist_ok=True)

    # Loop over each image in results and save detection annotations
    for i in range(len(results.pandas().xywh)):
        filename = results.files[i]
        # Remove .jpg extension
        filename = filename[:-4]
        # drop class column and reorder for globox expected order
        df = results.pandas().xywh[i][
            ["name", "xcenter", "ycenter", "width", "height", "confidence"]
        ]

        output_file = f"detections/{filename}.txt"
        df.to_csv(output_file, sep=" ", index=False, header=False)

    # Not using from_yolo_v5 as it doesn't allow to override relative=False
    # See https://github.com/laclouis5/globox/discussions/48
    # and https://github.com/laclouis5/globox/issues/49
    predictions = AnnotationSet.from_txt(
        folder="./detections",
        image_folder="./dataset/test",
        box_format=BoxFormat.XYWH,
        relative=False,
        separator=None,
        conf_last=True,
    )
    evaluator = COCOEvaluator(ground_truths=gt, predictions=predictions)
    evaluator.show_summary()
    evaluator.save_csv("evaluation.csv")


if __name__ == "__main__":
    model_path = sys.argv[1]
    gt_label_path = sys.argv[2]
    model_evaluate(model_path, gt_label_path)
