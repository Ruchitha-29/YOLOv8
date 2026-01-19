import os
import cv2
import pickle
import numpy as np

from insightface.app import FaceAnalysis
from adaface import AdaFace

KNOWN_FACES_DIR = "known_faces"
CACHE_DIR = "face_cache"

ARC_CACHE = os.path.join(CACHE_DIR, "arcface.pkl")
ADA_CACHE = os.path.join(CACHE_DIR, "adaface.pkl")

def ensure_cache():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

# ---------- Load or build ArcFace embeddings ----------
def load_arcface_embeddings():
    if os.path.exists(ARC_CACHE):
        print("[ARC] Loading ArcFace cache...")
        with open(ARC_CACHE, "rb") as f:
            return pickle.load(f)
    return None

def build_arcface_embeddings():
    print("[ARC] Building ArcFace embeddings...")
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(320,320))

    embeddings = []
    names = []

    for person in sorted(os.listdir(KNOWN_FACES_DIR)):
        ppath = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(ppath): continue

        for imgname in os.listdir(ppath):
            if imgname.lower().endswith(("jpg","jpeg","png")):
                ipath = os.path.join(ppath, imgname)
                img = cv2.imread(ipath)
                if img is None: continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = app.get(rgb)
                if not faces:
                    print("[ARC] No face in", imgname)
                    continue

                emb = faces[0].embedding.astype(np.float32)
                embeddings.append(emb)
                names.append(person)
                print("[ARC] Embedded:", person, imgname)

    data = {"embeddings": np.array(embeddings), "names": np.array(names)}
    with open(ARC_CACHE, "wb") as f:
        pickle.dump(data, f)

    print("[ARC] Saved cache.")
    return data

# ---------- Load or build AdaFace embeddings ----------
def load_adaface_embeddings():
    if os.path.exists(ADA_CACHE):
        print("[ADA] Loading AdaFace cache...")
        with open(ADA_CACHE, "rb") as f:
            return pickle.load(f)
    return None

def build_adaface_embeddings():
    print("[ADA] Building AdaFace embeddings...")
    model = AdaFace(pretrained="webface4m")  # strongest pretrained model

    embeddings = []
    names = []

    for person in sorted(os.listdir(KNOWN_FACES_DIR)):
        ppath = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(ppath): continue

        for imgname in os.listdir(ppath):
            if imgname.lower().endswith(("jpg","jpeg","png")):
                ipath = os.path.join(ppath, imgname)
                img = cv2.imread(ipath)
                if img is None: continue

                emb = model.get_embedding(img)
                embeddings.append(emb)
                names.append(person)
                print("[ADA] Embedded:", person, imgname)

    data = {"embeddings": np.array(embeddings), "names": np.array(names)}
    with open(ADA_CACHE, "wb") as f:
        pickle.dump(data, f)

    print("[ADA] Saved cache.")
    return data

# ---------- Loader Called from security_cam.py ----------
def load_hybrid_embeddings():
    ensure_cache()

    arc = load_arcface_embeddings()
    if arc is None:
        arc = build_arcface_embeddings()

    ada = load_adaface_embeddings()
    if ada is None:
        ada = build_adaface_embeddings()

    return arc, ada
