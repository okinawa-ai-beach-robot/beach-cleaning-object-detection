import numpy as np
import cv2 as cv
import onnxruntime
import beachbot

import time

# def wrap_detection(input_image, output_data):
#     class_ids = []
#     confidences = []
#     boxes = []

#     rows = output_data.shape[0]
#     image_width, image_height, _ = input_image.shape

#     x_factor = 1
#     y_factor = 1

#     for r in range(rows):
#         row = output_data[r]
#         confidence = row[4]

#         if confidence >= 0.2:

#             classes_scores = row[5:]
#             class_id = np.argmax(classes_scores)

#             if classes_scores[class_id] > 0.25:
#                 confidences.append(confidence)

#                 class_ids.append(class_id)

#                 x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
#                 left = int((x - 0.5 * w) * x_factor)
#                 top = int((y - 0.5 * h) * y_factor)
#                 width = int(w * x_factor)
#                 height = int(h * y_factor)
#                 box = np.array([left, top, width, height])
#                 boxes.append(box)

#     indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

#     result_class_ids = []
#     result_confidences = []
#     result_boxes = []

#     for i in indexes:
#         result_confidences.append(confidences[i])
#         result_class_ids.append(class_ids[i])
#         result_boxes.append(boxes[i])

#     return result_class_ids, result_confidences, result_boxes


# def apply_model(location, inputs):
#     session = onnxruntime.InferenceSession(location)
#     if session is None:
#         raise ValueError("Failed to load the model")
#     inputs = np.swapaxes(np.swapaxes(inputs, 0, -1), -2, -1)[None, :, :, :] / 255
#     inputs = inputs.astype(np.float16)
#     prediction = session.run(None, {"images": inputs})
#     return prediction


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
    #cam = beachbot.sensors.JetsonGstCameraNative()
    cam = beachbot.sensors.JetsonCsiCameraOpenCV()
    viewer = beachbot.utils.ImageViewerMatplotlib
    detector = beachbot.ai.Yolo5Onnx("../Models/beachbot_yolov5s_beach-cleaning-object-detection__v2i__yolov5pytorch_1280/best.onnx")
    #opencv only supports fp32 in installed version!
    #detector = beachbot.ai.Yolo5OpenCV("../Models/beachbot_yolov5s_beach-cleaning-object-detection__v2i__yolov5pytorch_1280/best.onnx")
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

            print("process time is:", t_show-t_start, "s")
            print("(frame grab", t_capture-t_start, "s, reshaping", t_reshape-t_capture, "s, detect", t_detect-t_reshape, "s, draw", t_draw-t_detect, "s, show", t_show-t_draw,"s")
    except KeyboardInterrupt as ex:
        pass
    wnd.close()
    cam.stop()
    return 0


if __name__ == "__main__":
    main()
