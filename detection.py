# pylint: disable=C0103,too-many-locals

import math
from typing import List, Tuple, Any

import cv2
import numpy as np

CONF_THRESH = 0.45
IOU_THRESH = 0.45
MAX_DET = 1000
MAX_WH = 4096
MAX_NMS = 30000


def xywh2xyxy(
    x: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def nms(boxes, scores: List[float], iou_thres: float):
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        prev_idxs = idxs[:-1]
        last_idx = idxs[-1]

        xx1 = np.maximum(x1[last_idx], x1[prev_idxs])
        yy1 = np.maximum(y1[last_idx], y1[prev_idxs])
        xx2 = np.minimum(x2[last_idx], x2[prev_idxs])
        yy2 = np.minimum(y2[last_idx], y2[prev_idxs])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[prev_idxs]
        overlap_idxs = prev_idxs[overlap > iou_thres]

        group_idxs = np.concatenate(([last_idx], overlap_idxs))
        group_best_idx = group_idxs[np.argmax(scores[group_idxs])]
        pick.append(group_best_idx)

        idxs = np.array([x for x in idxs if x not in group_idxs])

    return pick


def non_max_suppression(
    predictions,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
):
    candidates = predictions[..., 4] > conf_thres

    output = []

    for candidate, prediction in zip(candidates, predictions):
        prediction = prediction[candidate]
        num_box = len(prediction)
        if num_box == 0:
            output.append([])
            continue

        prediction[:, 5:] *= prediction[:, 4:5]
        box = xywh2xyxy(prediction[:, :4])
        box_class_scores = prediction[:, 5:]
        classes = np.argmax(box_class_scores, axis=1).reshape(-1, 1)
        conf = np.max(box_class_scores, axis=1).reshape(-1, 1)
        prediction = np.concatenate((box, conf, classes), axis=1)

        if num_box > MAX_NMS:
            prediction = prediction[-np.argsort(prediction[:, 4])][:MAX_NMS]

        c = prediction[:, 5:6] * MAX_WH
        boxes, scores = prediction[:, :4] + c, prediction[:, 4]
        indexes = nms(boxes, scores, iou_thres)
        indexes = indexes[:max_det]
        output.append(prediction[indexes])

    return output


class Rect:
    # pylint: disable=R0913
    def __init__(self, x1, y1, x2, y2, score, class_):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.class_ = class_

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2

    @property
    def x25(self):
        return self.x1 + 0.25 * self.width

    @property
    def x75(self):
        return self.x1 + 0.75 * self.width

    @property
    def y25(self):
        return self.y1 + 0.25 * self.height

    @property
    def y75(self):
        return self.y1 + 0.75 * self.height

    @property
    def area(self):
        return self.width * self.height

    def __repr__(self):
        return f"({self.x1},{self.y1},{self.x2},{self.y2})"


def get_gray_images(images: List[Any]
) -> List[Any]:
    gray_images = []

    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_images.append(np.stack([gray_image] * 3, axis=-1))

    return gray_images


def get_boxes(
    detection_model, images: List[Any], min_box_width=None, min_box_height=None
) -> List[List[Rect]]:
    detect_images = []

    input_height, input_width = detection_model.get_inputs()[0].shape[2:4]

    for img in images:
        img_h, img_w = img.shape[:2]
        img = cv2.resize(img, (input_width, input_height))
        img = np.transpose(img[:,:,::-1], (2,0,1))
        detect_images.append(img.astype("float32") / 255)
    
    input_name = detection_model.get_inputs()[0].name
    output_name = detection_model.get_outputs()[0].name
    predictions = detection_model.run([output_name], {input_name: np.array(detect_images)})[0]
    predictions = predictions[:, np.newaxis]
    predictions = np.transpose(predictions, (0,1,3,2))
    predictions = np.concatenate((predictions, np.ones((1, 1, predictions.shape[2],1))), axis=-1)
    boxes_list = non_max_suppression(predictions, CONF_THRESH, IOU_THRESH, MAX_DET)

    results = []
    for img, boxes in zip(images, boxes_list):
        img_h, img_w = img.shape[:2]
      
        boxes = [
            Rect(
                max(0, int((x1 - 0.01*(x2-x1)) * img_w / input_width)),
                max(0, int((y1 - 0.01*(y2-y1)) * img_h / input_height)),
                min(img_w, int((x2 + 0.01* (x2-x1)) * img_w / input_width + 0.5)),
                min(img_h, int((y2 + 0.01* (y2-y1)) * img_h / input_height + 0.5)),
                score,
                class_,
            )
            for x1, y1, x2, y2, score, class_ in boxes
            if (
                min(x2 - x1, y2 - y1) >= (min_box_width or 0) and
                max(x2 - x1, y2 - y1) >= (min_box_height or 0)
            )
        ]

        results.append(boxes)

    return results


def rotate_point(pt, xc, yc, angle):
    dx = pt[0] - xc
    dy = pt[1] - yc
    cosa, sina = math.cos(angle), math.sin(angle)
    dx, dy = dx * cosa - dy * sina, dx * sina + dy * cosa
    return [int(dx + xc), int(dy + yc)]


def rotate_box(box: Rect, xc, yc, angle):
    box_xc = box.center_x
    box_yc = box.center_y
    box_w = box.width
    box_h = box.height
    box_xc, box_yc = rotate_point((box_xc, box_yc), xc, yc, angle)
    return Rect(
        box_xc - box_w // 2,
        box_yc - box_h // 2,
        box_xc + box_w // 2,
        box_yc + box_h // 2,
        box.score,
        box.class_,
    )


def get_entropy(yarr, input_height):
    n_bin = 50
    bin_counts = [0] * n_bin

    for y in yarr:
        bin_index = int(y * n_bin / input_height + 0.5)
        if 0 <= bin_index < n_bin:
            bin_counts[bin_index] += 1

    return -sum([p * math.log(p) for p in bin_counts if p > 0])


def get_skew_angle(boxes, input_width, input_height):
    xc = input_width / 2
    yc = input_height / 2
    best_angle = 0
    best_entropy = get_entropy([box.center_y for box in boxes], input_height)

    for n in range(100):
        angle = (50 - n) / 100
        yarr = [
            rotate_point((box.center_x, box.center_y), xc, yc, angle)[-1]
            for box in boxes
        ]
        entropy = get_entropy(yarr, input_height)
        if entropy < best_entropy:
            best_angle = angle
            best_entropy = entropy

    return best_angle
