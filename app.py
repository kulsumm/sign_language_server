from flask import Flask, request, jsonify
import json, os
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import mediapipe as mp
import cv2
import zipfile
import gdown  # make sure to add to requirements.txt

app = Flask(__name__)

# ---- Download & unzip dataset if not present ----
DATASET_DIR = "dataset"
JSON_DIR = os.path.join(DATASET_DIR, "json")
CSV_FILE = os.path.join(DATASET_DIR, "test.csv")
GDRIVE_FILE_ID = "1QVHWO2xCxfpo7L7v5fPeDKql-TyUyC3e"  # your Drive file ID
ZIP_PATH = "dataset_json.zip"

if not os.path.exists(JSON_DIR):
    print("Downloading dataset from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}", ZIP_PATH, quiet=False)
    print("Unzipping dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    os.remove(ZIP_PATH)
    print("Dataset ready!")

# ---- Load dataset features ----
dataset_features = {}
for video_folder in os.listdir(JSON_DIR):
    folder_path = os.path.join(JSON_DIR, video_folder)
    if not os.path.isdir(folder_path):
        continue
    frames = []
    for file in sorted(os.listdir(folder_path)):
        if file.endswith("_keypoints.json"):
            data = json.load(open(os.path.join(folder_path, file)))
            frame_vec = []
            p = data['people'][0]
            for k in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
                frame_vec.extend(p.get(k, []))
            frames.append(frame_vec)
    dataset_features[video_folder] = np.array(frames)

# ---- Load glosses and translations ----
df = pd.read_csv(CSV_FILE)
glosses = dict(zip(df['id'], df['gloss']))
translations = dict(zip(df['id'], df['english']))

# ---- Initialize MediaPipe Holistic ----
mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    with mp_holistic as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = holistic.process(image)
            frame_vec = []
            for key in [res.pose_landmarks, res.face_landmarks, res.left_hand_landmarks, res.right_hand_landmarks]:
                if key:
                    for lm in key.landmark:
                        frame_vec.extend([lm.x, lm.y, lm.z])
                else:
                    frame_vec.extend([0]*21*3)  # default zeros if landmarks missing
            frames.append(frame_vec)
    cap.release()
    return np.array(frames)

def find_best_match(upload_features):
    best_vid = None
    best_dist = float('inf')
    for vid, features in dataset_features.items():
        dist, _ = fastdtw(upload_features, features, dist=euclidean)
        if dist < best_dist:
            best_dist = dist
            best_vid = vid
    return best_vid, best_dist

# ---- Flask endpoint ----
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['video']
    path = "temp_video.mp4"
    file.save(path)

    upload_features = extract_features(path)
    best_vid, dist = find_best_match(upload_features)

    threshold = 100  # tune based on your dataset
    if dist < threshold:
        predicted_gloss = glosses.get(best_vid, "Unknown")
        predicted_translation = translations.get(best_vid, "Unknown")
    else:
        predicted_gloss = "LLM fallback"
        predicted_translation = "LLM fallback"

    os.remove(path)  # clean up temp file

    return jsonify({
        "predicted_gloss": predicted_gloss,
        "predicted_translation": predicted_translation,
        "best_match_id": best_vid,
        "dtw_distance": dist
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1


