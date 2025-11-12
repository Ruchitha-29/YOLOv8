import face_recognition
import os
from typing import List, Any, Tuple

# Path to the parent folder containing the person-specific subfolders
FACES_DIR = "known_faces"

def load_known_faces() -> Tuple[List[Any], List[str]]:
    """
    Loads images from nested subfolders using os.walk.
    The subfolder's name is used as the person's name (label).
    """
    known_face_encodings = []
    known_face_names = []
    
    print("\n--- 🧑‍💻 Loading Authorized Faces (Recursive Scan) ---")
    
    # Ensure the directory exists
    if not os.path.isdir(FACES_DIR):
        print(f"Error: Directory '{FACES_DIR}' not found. Create it and add faces.")
        return known_face_encodings, known_face_names

    # Use os.walk to traverse all directories and subdirectories
    for root, dirs, files in os.walk(FACES_DIR):
        
        # Check if we are inside a person's subfolder (not the top 'known_faces' folder)
        if root != FACES_DIR:
            # The person's name is the name of the subfolder
            person_name = os.path.basename(root).capitalize() 

            for filename in files:
                # Check for common image file extensions
                if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    
                    # 1. Construct the full path to the image
                    path = os.path.join(root, filename)
                    
                    try:
                        # 2. Load the image
                        image = face_recognition.load_image_file(path)
                        
                        # 3. Find face locations and encode the first face found
                        face_locations = face_recognition.face_locations(image)
                        
                        if face_locations:
                            # Get the 128-dimensional face encoding
                            encoding = face_recognition.face_encodings(image, face_locations)[0]
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                            print(f"  ✅ Encoded: {person_name} from {path}")
                        else:
                            # This warning is crucial for debugging image quality
                            print(f"  ❌ Warning: No face found in {path}. Skipping image.")
                    
                    except Exception as e:
                        # Catch file reading or encoding errors without crashing
                        print(f"  ⚠ Error loading image {path}: {e}")

    return known_face_encodings, known_face_names

if __name__ == '_main_':
    encodings, names = load_known_faces()
    print(f"\nDatabase Load Complete. Found {len(encodings)} total encodings for {len(set(names))} unique people.")