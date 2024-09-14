import cv2
import numpy as np
import onnxruntime
import sys
from detection import get_boxes
from text_recognition import extract_texts

from flask import Flask, request, render_template, redirect
app = Flask(__name__)


plate_detect_model = onnxruntime.InferenceSession('plate.onnx', None)
char_detect_model = onnxruntime.InferenceSession('char.onnx', None)
ocr_model = onnxruntime.InferenceSession('ocr.onnx', None)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/extract-text", methods=['POST'])
def extract_text_api():
    img_file = request.files.get('img_file')
    
    if img_file and img_file.filename:
        try:
            img_path = 'static/' +  img_file.filename
            img_file.save(img_path)
            
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]

            if img_w/img_h > 4/3:
                crop_x = (img_w - 4*img_h//3)//2
                img = img[:, crop_x:img_w-crop_x]

            boxes = get_boxes(plate_detect_model, [img])[0]
            scale = img_h / 480

            for box in boxes:
                cv2.rectangle(img, (box.x1,box.y1),(box.x2,box.y2), (0,0,255), int(scale+0.5))
                plate_img = img[box.y1:box.y2,box.x1:box.x2]
                plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_img = np.stack([plate_img]*3, axis=-1)
                cv2.imwrite('plate.png', plate_img)
                text = extract_texts(char_detect_model, ocr_model, [plate_img])[0]
                cv2.putText(img, text.replace('\n', '-'), 
                    (box.x1, box.y2 + int(16 * scale)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    scale,
                    (0,255,0),
                    int(scale+0.5),
                    2
                )
            
            cv2.imwrite(img_path, img)
            return redirect('/' + img_path)

        except Exception as e:
            print(e)
            return str(e)
    else:
        return 'No file uploaded'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
