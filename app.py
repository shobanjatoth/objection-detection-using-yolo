import os
import cv2
import torch
import uuid
from flask import Flask, render_template, request, redirect, flash
from paddleocr import PaddleOCR
from ultralytics import YOLO
import numpy as np
import re

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

model = YOLO("best.pt")  # Your trained YOLO model
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Regex pattern for valid license plate text
plate_pattern = re.compile(r'^[A-Z0-9]{4,10}$')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image processing function
def process_image(image_path):
    img = cv2.imread(image_path)
    results = model(image_path)
    detected_texts = set()

    for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box.tolist())
        confidence = float(conf)
        plate_img = img[y1:y2, x1:x2]

        if plate_img.size == 0:
            continue

        gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        ocr_result = ocr.ocr(gray_plate, cls=True)

        if ocr_result:
            for res in ocr_result:
                for line in res:
                    text = line[1][0].strip().replace(" ", "").upper()
                    score = line[1][1]
                    if plate_pattern.match(text):
                        detected_texts.add(text)

                        # Draw bounding box with label
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{text} ({score:.2f})"
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(img, (x1, y1 - 25), (x1 + text_width, y1), (0, 255, 0), -1)
                        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    processed_image_path = os.path.join(PROCESSED_FOLDER, f"processed_{uuid.uuid4().hex}.jpg")
    cv2.imwrite(processed_image_path, img)
    return processed_image_path, list(detected_texts)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[-1]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            processed_file, ocr_results = process_image(file_path)
            return render_template("result.html",
                                   uploaded_file=filename,
                                   processed_file=os.path.basename(processed_file),
                                   ocr_results=ocr_results)
        else:
            flash("Only image files are allowed (png, jpg, jpeg, bmp). Video files like .mp4 are not supported.")
            return redirect(request.url)
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    
