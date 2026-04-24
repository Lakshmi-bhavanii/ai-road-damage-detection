from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import csv
import requests
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.cm as cm
from ultralytics import YOLO
from datetime import datetime
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= LOAD MODELS =================
cnn_model  = load_model('CNN.model')
yolo_model = YOLO("best.pt")

class_names = ["Moderate", "Poor", "Severe"]
IMG_SIZE    = 50

# ================= GPS =================
def get_gps():
    try:
        data = requests.get("http://ip-api.com/json/", timeout=5).json()
        lat  = data.get("lat") or 0
        lon  = data.get("lon") or 0
        city = data.get("city") or "Unknown"
        return float(lat), float(lon), str(city)
    except:
        return 0.0, 0.0, "Offline"

# ================= PREPROCESS =================
def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ================= DAMAGE % =================
def calculate_damage(boxes, shape):
    total = shape[0] * shape[1]
    pothole_area = 0
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        pothole_area += (x2 - x1) * (y2 - y1)
    return min((pothole_area / total) * 100, 100)

# ================= YOLO =================
def detect_pothole(img):
    r = yolo_model(img, conf=0.5, verbose=False)
    boxes = r[0].boxes
    annotated = r[0].plot()
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 229, 255), 2)
            cv2.putText(annotated, "Pothole", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2)
        return True, annotated, boxes
    return False, img, None

# ================= GRADCAM =================
def gradcam(img_tensor):
    last_conv = [l.name for l in cnn_model.layers if "conv" in l.name][-1]
    grad_model = tf.keras.models.Model(
        [cnn_model.inputs],
        [cnn_model.get_layer(last_conv).output, cnn_model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        loss = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(np.array(heatmap), 0)
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap

def overlay_heatmap(heatmap, img):
    h_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    h_uint8 = np.uint8(255 * h_resized)
    color = cm.jet(h_uint8)[:, :, :3]
    color = np.uint8(255 * color)
    return cv2.addWeighted(img, 0.6, color, 0.4, 0)

# ================= CSV =================
def save_csv(data):
    f_exist = os.path.isfile("road_data.csv")
    with open("road_data.csv", "a", newline="") as f:
        w = csv.writer(f)
        if not f_exist:
            w.writerow(["Image", "City", "Lat", "Lon", "Severity", "Conf", "Damage", "Timestamp"])
        w.writerow(data)

# ================= IMG -> BASE64 =================
def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode("utf-8")

# ================= PROCESS =================
def process_image(path, filename):
    img = cv2.imread(path)
    if img is None:
        return None, "Could not read image"
    original = img.copy()

    pothole, det_img, boxes = detect_pothole(img)

    if not pothole:
        return {
            "filename": filename,
            "pothole": False,
            "original_b64": img_to_b64(original),
            "message": "No pothole detected"
        }, None

    p = preprocess(img)
    preds = cnn_model.predict(p, verbose=0)
    idx   = int(np.argmax(preds))
    label = class_names[idx]
    conf  = float(np.max(preds))

    damage = calculate_damage(boxes, img.shape)
    lat, lon, city = get_gps()

    heatmap = gradcam(p)
    overlay = overlay_heatmap(heatmap, original)

    heatmap_vis = np.uint8(255 * cv2.resize(heatmap, (original.shape[1], original.shape[0])))
    heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_csv([filename, city, lat, lon, label, round(conf * 100, 2), round(damage, 2), timestamp])

    name_no_ext = os.path.splitext(filename)[0]
    cv2.imwrite(f"static/output/{name_no_ext}_det.jpg", det_img)
    cv2.imwrite(f"static/output/{name_no_ext}_overlay.jpg", overlay)
    cv2.imwrite(f"static/output/{name_no_ext}_heatmap.jpg", heatmap_color)

    return {
        "filename":    filename,
        "pothole":     True,
        "severity":    label,
        "confidence":  round(conf * 100, 1),
        "damage":      round(damage, 2),
        "city":        city,
        "lat":         lat,
        "lon":         lon,
        "timestamp":   timestamp,
        "original_b64":  img_to_b64(original),
        "detected_b64":  img_to_b64(det_img),
        "heatmap_b64":   img_to_b64(heatmap_color),
        "overlay_b64":   img_to_b64(overlay),
        "det_path":    f"/static/output/{name_no_ext}_det.jpg",
        "overlay_path": f"/static/output/{name_no_ext}_overlay.jpg",
        "heatmap_path": f"/static/output/{name_no_ext}_heatmap.jpg",
    }, None

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    results = []
    for f in files:
        if f.filename == "":
            continue
        # Split on both forward and back slashes to strip subfolder from webkitdirectory
        import re
        safe_name = re.split(r"[/\\]", f.filename)[-1]
        ext = os.path.splitext(safe_name)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        f.save(save_path)
        result, err = process_image(save_path, safe_name)
        if err:
            results.append({"filename": safe_name, "error": str(err)})
        elif result:
            # Sanitize: remove None keys, convert None values to empty string
            clean = {str(k): (v if v is not None else "") for k, v in result.items() if k is not None}
            results.append(clean)

    return jsonify({"results": results})

@app.route("/analyze_folder", methods=["POST"])
def analyze_folder():
    data   = request.get_json()
    folder = data.get("folder_path", "").strip()
    if not folder:
        return jsonify({"error": "No folder path provided"}), 400
    if not os.path.isdir(folder):
        return jsonify({"error": f"Folder not found: {folder}"}), 400
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [fn for fn in os.listdir(folder) if os.path.splitext(fn)[1].lower() in exts]
    if not images:
        return jsonify({"error": "No image files found in that folder"}), 400
    results = []
    for fname in images:
        path = os.path.join(folder, fname)
        result, err = process_image(path, fname)
        if err:
            results.append({"filename": fname, "error": err})
        elif result:
            results.append(result)
    return jsonify({"results": results, "total": len(images)})

@app.route("/history")
def history():
    rows = []
    if os.path.isfile("road_data.csv"):
        with open("road_data.csv", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
    return jsonify(rows)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

# Already written above — patching inline below