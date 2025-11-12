import cv2
import face_recognition
from ultralytics import YOLO
import numpy as np
import time
import serial 
from playsound import playsound 
import os 
import webbrowser 
# Removed the urllib.parse import as we no longer generate the file:/// URL

# Import the face loader function (assuming this is the recursive version)
from face_db_loader import load_known_faces

# --- Configuration Constants ---
WEBCAM_INDEX = 0 	 	 	
YOLO_MODEL = 'yolov8n.pt' 	
CONFIDENCE_THRESHOLD = 0.5 	
FACE_MATCH_TOLERANCE = 0.6 	
PERSON_CLASS_ID = 0 	 	
PADDING = 20 	 	 	 	
ALARM_SOUND_FILE = "alarm.mp3" 
LOG_FOLDER = "alarm_logs" 	
# Removed HTML_TEMPLATE_FILE and HTML_OUTPUT_FILE constants
ARDUINO_PORT = 'COM3' 	 	
BAUD_RATE = 9600
ALARM_COOLDOWN_SECONDS = 15 

# --- NEW CONSTANT: URL for the Flask Dashboard ---
DASHBOARD_URL = 'http://127.0.0.1:5000/' 


# --- Initial Setup ---
print("\n--- 🚨 Smart Security System Initialization ---")
# ... (Lines 50-79: Setup remains unchanged) ...
# A. Load the Face Database
known_face_encodings, known_face_names = load_known_faces()
if not known_face_encodings:
    print("\nFATAL ERROR: Face database is empty. Add images to the 'known_faces' folder.")
    exit()

# B. Load the YOLO model
try:
    model = YOLO(YOLO_MODEL)
    print(f"Model {YOLO_MODEL} loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load YOLO model: {e}")
    exit()

# C. Initialize Serial Connection
try:
    ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=0.1)
    print(f"Serial port {ARDUINO_PORT} opened successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not open serial port {ARDUINO_PORT}. Check port name or if it's open elsewhere.")
    print(e)
    exit()
# D. Global state tracker and camera object
is_active = False 
cap = None 	 
is_window_open = False 
last_alarm_time = 0.0 

print("--- System Ready. Waiting for motion signal from Arduino. Press 'q' to stop. ---")

# --- Function to Start and Stop Camera ---
def handle_camera_state(activate):
# ... (Lines 83-107: Function remains unchanged) ...
    global cap, is_active, is_window_open
    
    if activate and not is_active:
        print("Camera ON: Motion detected. Starting video feed...")
        cap = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW) 
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return False
        is_active = True
        is_window_open = True
        return True

    elif not activate and is_active:
        print("Camera OFF: Motion stopped. Deactivating video feed.")
        if cap:
            cap.release()
            cv2.destroyAllWindows() 
        cap = None
        is_active = False
        is_window_open = False
        return True
    
    return False

# --- Main Serial and Camera Loop ---
while True:
# ... (Lines 111-177: Main loop until unauthorized person detected remains unchanged) ...
    # 1. Check Serial Port for commands ('A' or 'D')
    try:
        if ser.in_waiting > 0:
            command = ser.read().decode('utf-8').strip()
            if command == 'A':
                handle_camera_state(True)
            elif command == 'D':
                handle_camera_state(False)
        
    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
        time.sleep(0.5)
    
    # 2. Run Detection ONLY if the camera is active
    if is_active and cap:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Restarting camera process.")
            handle_camera_state(False)
            continue

        main_status = "AUTHORIZED ACCESS - ANALYZING..."
        alarm_color = (0, 255, 0) # BGR: Green

        # 3. YOLOv8 Detection and Face Recognition Logic
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0].item())
                
                if cls == PERSON_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Define the person_crop (with padding)
                    crop_y1 = max(0, y1 - PADDING)
                    crop_x1 = max(0, x1 - PADDING)
                    crop_y2 = min(frame.shape[0], y2 + PADDING)
                    crop_x2 = min(frame.shape[1], x2 + PADDING)
                    person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    # --- FIX: Color Conversion (BGR -> RGB) and Error Handling ---
                    face_encodings = []
                    
                    try:
                        rgb_person_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_person_crop)
                        
                        if face_locations:
                            face_encodings = face_recognition.face_encodings(rgb_person_crop, face_locations)
                            
                    except Exception as e:
                        print(f"RUNTIME WARNING: Face encoding failed: {e}") 
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, "Encoding Error", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                        continue

                    # --- Comparison Logic ---
                    if face_encodings:
                        matches = face_recognition.compare_faces(
                            known_face_encodings, 
                            face_encodings[0], 
                            tolerance=FACE_MATCH_TOLERANCE
                        )

                        if True in matches:
                            # Authorized Person
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
                            best_match_index = np.argmin(face_distances)
                            name = known_face_names[best_match_index]
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            # --- UNAUTHORIZED ALARM TRIGGER & ALERT (MODIFIED) ---
                            main_status = "!!! UNAUTHORIZED ACCESS !!!"
                            alarm_color = (0, 0, 255) # BGR: Red
                            
                            current_time = time.time()
                            
                            # 1. Enforce Cooldown for Image Capture and Alert
                            if current_time - last_alarm_time > ALARM_COOLDOWN_SECONDS:
                                
                                # 1a. SAVE THE IMAGE (LOGGING)
                                timestamp = time.strftime("%Y%m%d-%H%M%S")
                                log_filename = f"UNAUTHORIZED_{timestamp}.jpg"
                                log_filepath = os.path.join(LOG_FOLDER, log_filename)
                                
                                try:
                                    if not os.path.exists(LOG_FOLDER):
                                        os.makedirs(LOG_FOLDER)
                                    cv2.imwrite(log_filepath, frame) 
                                    print(f"EVIDENCE SAVED: {log_filepath}")
                                    
                                    # 1b. OPEN THE FLASK DASHBOARD URL IN THE SAME TAB
                                    # This triggers the web page to load and fetch ALL the new images
                                    webbrowser.open(DASHBOARD_URL, new=0, autoraise=True)
                                    
                                    # 1c. Update the Cooldown Timer
                                    last_alarm_time = current_time 
                                    
                                except Exception as e:
                                    print(f"ERROR saving image or launching alert: {e}")

                                # 2. Trigger the sound alarm! (Happens when cooldown is over)
                                try:
                                    playsound(ALARM_SOUND_FILE, block=False) 
                                except Exception as e:
                                    print(f"ALARM SOUND FAILED: {e}.")

                                print(f"ALARM: Unauthorized person detected and dashboard updated at {time.strftime('%H:%M:%S')}!")

                            # 3. Update Visuals (Happens every frame an unauthorized person is detected)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), alarm_color, 2)
                            cv2.putText(frame, "UNAUTHORIZED", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, alarm_color, 2)
                            
                    else:
                        # Person detected, but no recognizable face found
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        cv2.putText(frame, "Face Not Found", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

                else:
                    # For non-person objects, draw the default YOLO box
                    frame = r.plot(conf=True, labels=True)

        # 4. Display Main Status/Alarm Banner
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), alarm_color, -1)
        cv2.putText(frame, main_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 5. Show the frame
        cv2.imshow("Smart Security System", frame)

    # 6. Handle Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Pause when idle (not active) to save CPU
    if not is_active:
        time.sleep(0.1)

# --- Final Cleanup ---
handle_camera_state(False) 
ser.close()
cv2.destroyAllWindows()
print("\nSystem Shutdown Complete.")