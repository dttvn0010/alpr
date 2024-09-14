from typing import List, Dict, Tuple, Any

import cv2
import numpy as np

from detection import Rect, get_boxes, rotate_box, get_skew_angle

CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']

CHAR_WIDTH = 32
CHAR_HEIGHT = 44
MIN_CHAR_WIDTH = 3
MIN_CHAR_HEIGHT = 16

class CharBox:
    def __init__(self, box: Rect, char: str):
        self.bound = box
        self.char = char

def get_bound(char_boxes: List[CharBox]):
    x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0

    for char_box in char_boxes:
        char_bound = char_box.bound
        x1 = min(x1, char_bound.x1)
        y1 = min(y1, char_bound.y1)
        x2 = max(x2, char_bound.x2)
        y2 = max(y2, char_bound.y2)

    return Rect(x1, y1, x2, y2, 1.0, 0)


def get_text(char_boxes: List[CharBox]) -> str:
    char_boxes_by_lines: List[List[CharBox]] = []

    for char_box in char_boxes:
        char_bound = char_box.bound
        for line in char_boxes_by_lines:
            bound = get_bound(line)
            if bound.y25 < char_bound.y75 and bound.y75 > char_bound.y25:
                line.append(char_box)
                break
        else:
            char_boxes_by_lines.append([char_box])

    indexes = np.argsort([get_bound(line).center_y for line in char_boxes_by_lines])
    
    lines = []
    for index in indexes:
        line_char_boxes = sorted(char_boxes_by_lines[index], key=lambda x : x.bound.center_x)
        lines.append(''.join([cb.char for cb in line_char_boxes]))
    
    correct_func1 = lambda x : {'0': 'D', '2': 'Z', '8': 'B', '5': 'S', '7': 'Z'}.get(x,x)
    correct_func2 = lambda x : {'D': '0', 'Z': '2', 'B': '8', 'S': '5', 'T': '1'}.get(x,x)
    lines = [l for l in lines if len(l) >= 2]
    
    if len(lines) > 0 and len(lines[0]) == 4:
        tmp = ''
        for i,ch in enumerate(lines[0]):
            if i != 2:
                tmp += correct_func2(ch)
            else:
                tmp += correct_func1(ch)

        lines[0] = tmp
        
    if len(lines) > 1:
        lines[1] = ''.join(map(correct_func2, list(lines[1])))
        
    return '\n'.join(lines)


def extract_texts(
    detection_model,
    ocr_model,
    images: List[Any],
) -> List[str]:

    boxes_list = get_boxes(detection_model, images, MIN_CHAR_WIDTH, MIN_CHAR_HEIGHT)
    text_images = []

    for image, boxes in zip(images, boxes_list):
        for box in boxes:
            text_image = cv2.resize(
                cv2.cvtColor(
                    image[box.y1 : box.y2, box.x1 : box.x2],
                    cv2.COLOR_BGR2GRAY
                ),
                (CHAR_WIDTH, CHAR_HEIGHT),
            )
            text_image_mean = np.mean(text_image)
            text_image_std = np.std(text_image)
            text_images.append(
                (text_image - text_image_mean) / max(36.0, 3 * text_image_std)
            )

    if len(text_images) > 0:
        text_images = np.array(text_images, dtype='float32')
        text_images = np.expand_dims(text_images, axis=-1)
        input_name = ocr_model.get_inputs()[0].name
        output_name = ocr_model.get_outputs()[0].name
        ocr_preds = ocr_model.run([output_name], {input_name: text_images})[0]
    else:
        ocr_preds = []

    results = []
    offset = 0

    for image, boxes in zip(images, boxes_list):
        img_h, img_w = image.shape[:2]
        angle = get_skew_angle(boxes, img_w, img_h)
        box_count = len(boxes)

        if box_count > 0:
            boxes = [rotate_box(box, img_w / 2, img_h / 2, angle) for box in boxes]
            char_boxes = [
                CharBox(box, CHARACTERS[np.argmax(ocr_score)])
                for (box, ocr_score) in zip(
                    boxes, ocr_preds[offset : offset + box_count]
                )
            ]

            results.append(get_text(char_boxes))
            offset += box_count
        else:
            results.append('')

    return results
