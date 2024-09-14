import cv2
import numpy as np
import onnxruntime
import sys
from detection import get_boxes
from text_recognition import extract_texts

plate_detect_model = onnxruntime.InferenceSession('plate.onnx', None)
char_detect_model = onnxruntime.InferenceSession('char.onnx', None)
ocr_model = onnxruntime.InferenceSession('ocr.onnx', None)

img = cv2.imread(sys.argv[1])
img_h, img_w = img.shape[:2]

if img_w/img_h > 4/3:
    crop_x = (img_w - 4*img_h//3)//2
    img = img[:, crop_x:img_w-crop_x]

boxes = get_boxes(plate_detect_model, [img])[0]
print(img.shape, boxes)

for box in boxes:
    plate_img = img[box.y1:box.y2,box.x1:box.x2]
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_img = np.stack([plate_img]*3, axis=-1)
    text = extract_texts(char_detect_model, ocr_model, [plate_img])[0]
    print(text)
    scale = img_h / 480
    
    cv2.putText(img, text.replace('\n', '-'), 
        (box.x1, box.y2 + int(16 * scale)), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        scale,
        (0,0,255),
        int(scale),
        2
    )

cv2.imwrite('out.jpg', img)
