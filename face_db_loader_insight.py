import os
import numpy as np
import cv2
from insightface.app import FaceAnalysis

KNOWN_FACES_DIR = "known_faces"
CACHE_DIR = "face_cache"
EMB_FILE = os.path.join(CACHE_DIR, "embeddings.npy")
NAME_FILE = os.path.join(CACHE_DIR, "names.npy")


def ensure_cache_folder():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)


def load_cached_embeddings():
    """Loads embeddings from cache if available."""
    if os.path.exists(EMB_FILE) and os.path.exists(NAME_FILE):
        print("[CACHE] Loading embeddings from face_cache/ ...")
        embeddings = np.load(EMB_FILE)
        names = np.load(NAME_FILE)
        return embeddings, names
    return None, None


def generate_embeddings():
    """Build embeddings using InsightFace and save to cache."""
    print("[CACHE] Cache not found → Generating new embeddings...")

    fa = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    fa.prepare(ctx_id=-1, det_size=(320, 320))

    embeddings = []
    names = []

    for person in sorted(os.listdir(KNOWN_FACES_DIR)):
        person_path = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in sorted(os.listdir(person_path)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = fa.get(rgb)
            if not faces:
                print(f"[WARN] No face in {img_path}")
                continue

            emb = faces[0].embedding.astype(np.float32)
            embeddings.append(emb)
            names.append(person)

            print(f"[OK] Embedded: {person}/{img_name}")

    embeddings = np.array(embeddings)
    names = np.array(names)

    ensure_cache_folder()
    np.save(EMB_FILE, embeddings)
    np.save(NAME_FILE, names)

    print("[CACHE] Embeddings saved successfully.")

    return embeddings, names


def load_or_create_embeddings():
    ensure_cache_folder()

    emb, names = load_cached_embeddings()
    if emb is not None:
        return emb, names

    return generate_embeddings()
