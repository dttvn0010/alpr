import cv2
import onnxruntime
import sys
from detection import get_boxes

model = onnxruntime.InferenceSession('char.onnx', None)

img = cv2.imread(sys.argv[1])
boxes = get_boxes(model, [img])[0]

for box in boxes:
    x1,y1,x2,y2 = map(int,[box.x1,box.y1,box.x2,box.y2])
    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),1)

cv2.imwrite('out.jpg',img)
