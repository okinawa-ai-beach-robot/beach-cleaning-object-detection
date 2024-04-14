import numpy as np
import cv2 as cv
import onnxruntime
import beachbot

import time


def set_resolution(intended_x, intended_y, image):
    h = image.shape[0]
    w = image.shape[1]
    ratio_w = intended_x / w
    ratio_h = intended_y / h
    maxratio = max(ratio_w, ratio_h)
    image = cv.resize(image, (int(w * maxratio), int(h * maxratio)))
    crop_w = image.shape[1] - intended_x
    crop_h = image.shape[0] - intended_y
    if crop_h > 0:
        image = image[crop_h // 2 : -crop_h // 2]
    if crop_w > 0:
        image = image[:, crop_w // 2 : -crop_w // 2]
    return image


def draw_box(class_ids, confidences, boxes, image, config):
    colors = [
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
        (255, 0, 0),
    ]
    for classid, confidence, box in zip(class_ids, confidences, boxes):
        if confidence >= config:
            color = colors[int(classid) % len(colors)]
            cv.rectangle(image, box, color, 2)
    img2 = image[:, :, ::-1]
    return img2


def main():
    cam = beachbot.sensors.JetsonGstCameraNative()
    #cam = beachbot.sensors.JetsonCsiCameraOpenCV()
    viewer = beachbot.utils.ImageViewerMatplotlib
    detector = beachbot.ai.Yolo5Onnx(
        "../Models/beachbot_yolov5s_beach-cleaning-object-detection__v2i__yolov5pytorch_1280/best.onnx"
    )
    # opencv only supports fp32 in installed version!
    # detector = beachbot.ai.Yolo5OpenCV("../Models/beachbot_yolov5s_beach-cleaning-object-detection__v2i__yolov5pytorch_1280/best.onnx")
    wnd = viewer("Annotated Image")
    print("Starting detection loop...")
    try:
        while True:
            # capture the next image
            t_start = time.time()
            frame = cam.read()
            t_capture = time.time()

            if frame is None:  # timeout
                continue

            frame = detector.crop_and_scale_image(frame)
            t_reshape = time.time()
            class_ids, confidences, boxes = detector.apply_model(frame)
            t_detect = time.time()

            img2 = draw_box(class_ids, confidences, boxes, frame, 0.2)
            t_draw = time.time()

            wnd.show(img2)
            t_show = time.time()

            print("process time is:", t_show - t_start, "s")
            print(
                "(frame grab",
                t_capture - t_start,
                "s, reshaping",
                t_reshape - t_capture,
                "s, detect",
                t_detect - t_reshape,
                "s, draw",
                t_draw - t_detect,
                "s, show",
                t_show - t_draw,
                "s",
            )
    except KeyboardInterrupt as ex:
        pass
    wnd.close()
    cam.stop()
    return 0


if __name__ == "__main__":
    main()
