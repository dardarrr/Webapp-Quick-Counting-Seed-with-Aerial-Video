from flask import Flask, request, jsonify, render_template, Response
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import time
from sort import Sort
import json

app = Flask(__name__)

# Folder untuk upload, model, dan output
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

status_updates = []  # Global untuk menyimpan status


def generate_status_updates():
    """Fungsi untuk streaming status update ke client"""
    try:
        while True:
            if status_updates:
                message = status_updates.pop(0)
                yield f"data: {json.dumps(message)}\n\n"  # Format sebagai JSON
            time.sleep(1)
    except GeneratorExit:
        print("Client disconnected.")


@app.route('/')
def index():
    return render_template('awal.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/status')
def status():
    return Response(generate_status_updates(), mimetype='text/event-stream')


@app.route('/process-video', methods=['POST'])
def process_video():
    global status_updates
    status_updates = []  # Reset status updates

    # Ambil file dari form
    video_file = request.files['video']
    model_file = request.files['model']
    output_video_name = request.form['output_video']
    line_coords = list(map(int, request.form['line_coords'].split(',')))
    label_count = request.form['label_count']
    confidence_threshold = float(request.form['confidence_threshold'])
    bagi_slice = int(request.form['bagi_slice'])

    # Simpan video dan model ke folder yang sudah ditentukan
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    model_path = os.path.join(MODEL_FOLDER, model_file.filename)
    video_file.save(video_path)
    model_file.save(model_path)

    # Buat path untuk output video
    output_video_path = os.path.join(OUTPUT_FOLDER, output_video_name)

    # Inisialisasi video capture dan video writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status_updates.append({"step": "Cannot open the video file.", "completed": True})
        return jsonify({'error': 'Cannot open the video file.'}), 400

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # Inisialisasi tracker
    tracker = Sort(max_age=60, min_hits=2, iou_threshold=0.3)
    try:
        model = YOLO(model_path)
    except Exception as e:
        status_updates.append({"step": "The YOLO model cannot be loaded.", "completed": True})
        return jsonify({'error': 'The YOLO model cannot be loaded.'}), 500

    status_updates.append({"step": "Starting video processing...", "completed": False})
    start_process_time = time.perf_counter()
    total_count = 0
    counted_ids = set()

    def slice_image(image, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio):
        image_np = np.array(image)
        slices = []
        h, w, _ = image_np.shape
        step_h = int(slice_height * (1 - overlap_height_ratio))
        step_w = int(slice_width * (1 - overlap_width_ratio))
        for y in range(0, h, step_h):
            for x in range(0, w, step_w):
                y1 = y
                y2 = min(y + slice_height, h)
                x1 = x
                x2 = min(x + slice_width, w)
                slices.append({
                    'image': image_np[y1:y2, x1:x2],
                    'starting_pixel': (x1, y1)
                })
        return slices

    def non_max_suppression_fast(boxes, overlapThresh):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        score = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(score)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        return boxes[pick]

    

    frame_number = 0  # Tambahkan penghitung frame

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            status_updates.append({"step": f"Processing frames {frame_number}", "completed": False})

            # Mengubah frame menjadi gambar PIL untuk pemrosesan lebih lanjut
            image = Image.fromarray(frame)
            height, width = image.size
            slice_height = int(height / bagi_slice)
            slice_width = int(width / bagi_slice)

            # Memotong frame menjadi beberapa bagian
            status_updates.append({"step": f"Slicing the frame into {bagi_slice} parts", "completed": False})
            slice_image_result = slice_image(
                image=image,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            bboxes = []

            # Deteksi objek pada setiap potongan gambar
            status_updates.append({"step": "Detecting objects in each sliced image", "completed": False})
            for i, image_slice in enumerate(slice_image_result):
                window = image_slice['image']
                start_x, start_y = image_slice['starting_pixel']
                results = model(window, conf=confidence_threshold, verbose=False, iou=0.5)

                for result in results:
                    boxes = result.boxes
                    xyxy = boxes.xyxy.cpu().numpy()

                    if xyxy.size == 0:
                        continue

                    conf = boxes.conf.cpu().numpy()
                    class_id = boxes.cls.cpu().numpy()

                    for i in range(len(xyxy)):
                        x1, y1, x2, y2 = xyxy[i]
                        x1 += start_x
                        y1 += start_y
                        x2 += start_x
                        y2 += start_y
                        bboxes.append([int(x1), int(y1), int(x2), int(y2), conf[i]])

            # Non-Maximum Suppression (NMS) untuk menghapus bounding box yang berlebihan
            status_updates.append({"step": "Reassembling the image slices into a single frame", "completed": False})
            if len(bboxes) > 0:
                bboxes = non_max_suppression_fast(np.array(bboxes), overlapThresh=0.3)

            # Melakukan pelacakan objek
            status_updates.append({"step": "Tracking objects", "completed": False})
            tracked_objects = tracker.update(bboxes)

            # Menggambar bounding box dan menghitung objek yang melewati garis
            status_updates.append({"step": "Drawing bounding boxes and counting objects", "completed": False})
            for obj in tracked_objects:
                if not np.isnan(obj).any():
                    x1, y1, x2, y2, track_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    if line_coords[0] < cx < line_coords[2] and line_coords[1] < cy < line_coords[3]:
                        if track_id not in counted_ids:
                            total_count += 1
                            counted_ids.add(track_id)

            # Menambahkan teks dan garis pada frame
            text = f'{label_count}: {total_count}'
            font_scale = 2
            thickness = 3
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.putText(frame, text, (frame_width - text_width - 20, 50), font, font_scale, (0, 255, 0), thickness)
            cv2.line(frame, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), (0, 255, 0), 2)

            # Menyimpan frame ke output video
            out.write(frame)

    finally:
        cap.release()
        out.release()

    end_process_time = time.perf_counter()
    total_time = end_process_time - start_process_time
    status_updates.append({"step": f"Video processing completed in  {total_time:.2f} detik.", "completed": True})

    return jsonify({'output_video_path': output_video_path}), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
