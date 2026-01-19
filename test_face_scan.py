import face_recognition
import os

FACES_DIR = "known_faces"
count = 0
fails = []

for root, dirs, files in os.walk(FACES_DIR):
    for f in files:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            path = os.path.join(root, f)
            try:
                image = face_recognition.load_image_file(path)
                locs = face_recognition.face_locations(image)
                if locs:
                    count += 1
                else:
                    fails.append(path)
            except Exception as e:
                fails.append(path)

print("\n=== FACE SCAN REPORT ===")
print("Detected faces in:", count)
print("Failed to detect:", len(fails))
print("\nFiles with no detected faces:")
for f in fails:
    print(" -", f)
