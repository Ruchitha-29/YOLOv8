"""
Final security_cam.py — INSTANT RECOGNITION VERSION
----------------------------------------------------
Features:
 - YOLO face model (yolov8n-face.pt)
 - ArcFace (buffalo_l) recognition
 - Instant recognition → GREEN box (NO pending)
 - UNAUTHORIZED → RED box + dashboard popup
 - Face quality checks (blur / dark / bright)
 - Faster processing (frame skip, optimized transforms)
 - PIR support (Arduino)
"""

import os
import sys
import time
import json
import cv2
import numpy as np

# optional libs
try:
    import serial
except:
    serial = None

try:
    from ultralytics import YOLO
except Exception as e:
    print("[FATAL] ultralytics not installed:", e)
    sys.exit(1)

try:
    from playsound import playsound
except:
    playsound = None

try:
    from insightface.app import FaceAnalysis
except Exception as e:
    print("[FATAL] insightface not installed:", e)
    print("Install: pip install insightface onnxruntime")
    sys.exit(1)

try:
    import win32gui, win32con
    WINDOWS_UI = True
except:
    WINDOWS_UI = False


# ---------------- CONFIG ----------------
WEBCAM_INDEX = 0
YOLO_MODEL = "yolov8n-face.pt"

CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0

PADDING = 5
MIN_FACE_SIZE = 80

IDENTITY_THRESHOLD = 0.60

CAMERA_HOLD_SECONDS = 20
ALARM_COOLDOWN_SECONDS = 15

ARDUINO_PORT = "COM3"
BAUD_RATE = 9600

LOG_FOLDER = "alarm_logs"
LOG_FILE = os.path.join(LOG_FOLDER, "intrusion_log.json")

ALARM_SOUND_FILE = "alarm.mp3"

FIXED_LOCATION = "13.75918, 77.34644"

DASHBOARD_TITLES = ["Security Dashboard", "127.0.0.1:5000"]


# ---------------- HELPERS ----------------
def ensure_log_folder():
    os.makedirs(LOG_FOLDER, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump({"logs": []}, f, indent=4)


def save_intrusion_log(filename, timestamp, location):
    try:
        with open(LOG_FILE, "r") as f:
            data = json.load(f)
    except:
        data = {"logs": []}

    data["logs"].insert(0, {
        "filename": filename,
        "timestamp": timestamp,
        "location": location
    })
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)


def focus_dashboard():
    if not WINDOWS_UI:
        return

    def enumHandler(hwnd, result):
        title = win32gui.GetWindowText(hwnd)
        for key in DASHBOARD_TITLES:
            if key.lower() in title.lower():
                result.append(hwnd)

    found = []
    win32gui.EnumWindows(enumHandler, found)

    if found:
        try:
            hwnd = found[0]
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(0.05)
            win32gui.SetForegroundWindow(hwnd)
        except:
            pass


def l2_normalize_vec(x):
    x = x.astype(np.float32)
    return x / (np.linalg.norm(x) + 1e-10)


def is_face_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 60


def is_face_too_dark(img):
    return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) < 40


def is_face_too_bright(img):
    return np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) > 210


# ---------------- MAIN ----------------
def main():
    ensure_log_folder()

    print("[INIT] Loading YOLO...")
    model = YOLO(YOLO_MODEL)
    try:
        model.fuse()
    except:
        pass

    print("[INIT] Loading ArcFace (buffalo_l)...")
    fa = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
    fa.prepare(ctx_id=-1, det_size=(160, 160))

    print("[INIT] Loading known faces...")
    known_embs = []
    known_names = []

    base = "known_faces"
    if os.path.isdir(base):
        for person in sorted(os.listdir(base)):
            folder = os.path.join(base, person)
            if not os.path.isdir(folder): continue

            for img_name in sorted(os.listdir(folder)):
                if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                path = os.path.join(folder, img_name)
                img = cv2.imread(path)
                if img is None: continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                faces = fa.get(rgb)
                if not faces:
                    print("[WARN] No face in", path)
                    continue

                face_obj = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                emb = l2_normalize_vec(face_obj.embedding)
                known_embs.append(emb)
                known_names.append(person)
                print(f"[OK] Indexed {person}/{img_name}")

    known_embs = np.array(known_embs, dtype=np.float32) if known_embs else np.zeros((0,512), np.float32)

    # Serial / PIR
    ser = None
    if serial:
        try:
            ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
            print("[SERIAL] Using PIR on", ARDUINO_PORT)
        except:
            print("[WARN] Could not open PIR serial.")
            ser = None

    cap = None
    is_active = False
    hold_until = 0
    last_alarm = 0
    frame_count = 0

    print("[READY] System running...")

    while True:

        # ---------------- PIR Handling ----------------
        if ser and ser.in_waiting > 0:
            msg = ser.readline().decode(errors='ignore').strip()
            print("[SERIAL]", msg)

            if "A" in msg or "Motion" in msg:
                if not is_active:
                    cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    is_active = True
                    print("Camera ON")
                hold_until = time.time() + CAMERA_HOLD_SECONDS

            elif "D" in msg or "Stop" in msg:
                hold_until = time.time() + CAMERA_HOLD_SECONDS
                print("Motion stopped → Hold active")

        if is_active and time.time() > hold_until:
            print("Camera OFF")
            if cap: cap.release()
            try: cv2.destroyAllWindows()
            except: pass
            is_active = False
            continue

        if not is_active:
            time.sleep(0.05)
            continue

        # ---------------- READ FRAME ----------------
        ret, frame = cap.read()
        if not ret:
            continue

        # Speed improvement – process every 2nd frame
        frame_count += 1
        if frame_count % 2 != 0:
            continue

        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                crop = frame[max(0, y1-PADDING):y2+PADDING,
                             max(0, x1-PADDING):x2+PADDING]

                if crop.size == 0:
                    continue

                # FACE QUALITY FILTERS
                if is_face_blurry(crop):
                    continue
                if is_face_too_dark(crop):
                    continue
                if is_face_too_bright(crop):
                    continue

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                faces = fa.get(rgb_crop)
                if not faces:
                    continue

                face_obj = faces[0]
                emb = l2_normalize_vec(face_obj.embedding)

                # MATCHING
                recognized = False
                name = None

                if len(known_embs) > 0:
                    sims = known_embs @ emb
                    idx = int(np.argmax(sims))
                    dist = 1 - sims[idx]

                    if dist < IDENTITY_THRESHOLD:
                        recognized = True
                        name = known_names[idx]

                # ---------- DRAW + ALARM ----------
                if recognized:
                    # GREEN BOX IMMEDIATELY
                    cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 3)
                    cv2.putText(frame, name, (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0,255,0), 2)

                else:
                    # UNAUTHORIZED → RED + LOG + DASHBOARD POPUP
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),3)
                    cv2.putText(frame, "UNAUTHORIZED", (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0,0,255), 2)

                    now = time.time()
                    if now - last_alarm > ALARM_COOLDOWN_SECONDS:
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        filename = f"UNAUTHORIZED_{ts}.jpg"
                        filepath = os.path.join(LOG_FOLDER, filename)

                        cv2.imwrite(filepath, frame)
                        save_intrusion_log(filename,
                            time.strftime("%d/%m/%Y %H:%M:%S"),
                            FIXED_LOCATION)

                        # Always open dashboard
                        focus_dashboard()
                        os.system("start http://127.0.0.1:5000/")

                        if playsound:
                            try:
                                playsound(ALARM_SOUND_FILE, block=False)
                            except:
                                pass

                        last_alarm = now

        try:
            cv2.imshow("Smart Security System", frame)
        except:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cap: cap.release()
    cv2.destroyAllWindows()
    if ser:
        try: ser.close()
        except: pass


if __name__ == "__main__":
    main()