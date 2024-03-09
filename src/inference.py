import numpy as np
import cv2 as cv
import onnxruntime
import threading


class ThreadedCamera:
    def __init__(self, resolution=(1280, 720), camera_id=0, fps=15):
        self._stopped = False
        self._cap = cv.VideoCapture(camera_id)
        self._cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution[1])
        # read the first frame
        self._ret, self._frame = self._cap.read()
        threading.Thread(target=self.run, args=(), daemon=True).start()

    def get_size(self):
        return (int(self._cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(self._cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    def run(self):
        while not self._stopped:
            self._ret, self._frame = self._cap.read()
        self._cap.release()

    def read(self):
        return self._frame

    def stop(self):
        self._stopped = True


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape

    x_factor = 1
    y_factor = 1

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]

        if confidence >= 0.2:

            classes_scores = row[5:]
            _, _, _, max_indx = cv.minMaxLoc(classes_scores)
            class_id = max_indx[1]

            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def apply_model(location, inputs):
    session = onnxruntime.InferenceSession(location)
    if session is None:
        raise ValueError("Failed to load the model")
    inputs = np.swapaxes(np.swapaxes(inputs, 0, -1), -2, -1)[None, :, :, :] / 255
    inputs = inputs.astype(np.float32)
    prediction = session.run(None, {"images": inputs})
    return prediction


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
        image = image[crop_h // 2:-crop_h // 2]
    if crop_w > 0:
        image = image[:, crop_w // 2:-crop_w // 2]
    return image


def draw_box(class_ids, confidences, boxes, image, config):
    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 0), (255, 0, 0), (255, 0, 0),
              (255, 0, 0), (255, 0, 0), (255, 0, 0)]
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        if confidence >= config:
            color = colors[int(classid) % len(colors)]
            cv.rectangle(image, box, color, 2)
    img2 = image[:, :, ::-1]
    return img2


def main():
    cam = ThreadedCamera(camera_id=0)
    while (1):
        frame0 = cam.read()
        frame = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        k = cv.waitKey(10) & 0xFF
        if k == 27:
            break
        frame2 = set_resolution(160, 96, frame)
        prediction = apply_model("../Models/beachbot_yolov5s_beach-cleaning-object-detection__v1i__yolov5pytorch_1280/best.onnx", frame2)
        class_ids, confidences, boxes = wrap_detection(frame2, prediction[0][0])
        img2 = draw_box(class_ids, confidences, boxes, frame2, 0.2)
        cv.imshow('image',img2)
    cv.destroyAllWindows()
    cam.stop()
    return 0

if __name__ == '__main__':
    main()
