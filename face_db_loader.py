import os
import numpy as np
import cv2
import face_recognition
from pathlib import Path

FACES_DIR = "known_faces"
CACHE_DIR = "face_cache"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def encode_face(image_path):
    """Load + encode one face from file"""
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")

    if not locs:
        return None

    return face_recognition.face_encodings(rgb, locs)[0]

def load_known_faces():
    encodings = []
    names = []

    # --- 1. LOAD FROM CACHE (FAST) ---
    cache_files = list(Path(CACHE_DIR).glob("*.npy"))
    if cache_files:
        for f in cache_files:
            obj = np.load(f, allow_pickle=True).item()
            encodings.append(obj["encoding"])
            names.append(obj["name"])
        print(f"[FAST] Loaded {len(names)} encodings from cache.")
        return encodings, names

    # --- 2. FIRST RUN: BUILD CACHE ---
    print("[INIT] First run — Building face cache...")

    for person in os.listdir(FACES_DIR):
        person_folder = os.path.join(FACES_DIR, person)

        if not os.path.isdir(person_folder):
            continue

        for file in os.listdir(person_folder):
            if file.lower().endswith(("jpg","jpeg","png")):
                path = os.path.join(person_folder, file)
                enc = encode_face(path)

                if enc is not None:
                    encodings.append(enc)
                    names.append(person)

                    # save cache file
                    cache_path = os.path.join(CACHE_DIR, f"{person}_{file}.npy")
                    np.save(cache_path, {"name": person, "encoding": enc})

                    print("[CACHED]", person, file)

    print("Cache ready! Encoded:", len(names))
    return encodings, names
