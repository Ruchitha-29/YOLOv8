from flask import Flask, render_template, jsonify
import os
import glob
import time


# --- SERVER CONFIGURATION ---
LOG_FOLDER = "alarm_logs"

if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Initialize Flask, setting the alarm_logs folder as the static file directory.
# This allows images to be accessed directly via the URL /filename.jpg
app = Flask(__name__, static_folder=LOG_FOLDER, static_url_path='/alarm_logs')

# Flask is configured to look for templates in the 'templates' folder automatically.

@app.route('/')
def dashboard():
    """Renders the main dashboard page from the 'templates/dashboard.html' file."""
    # The dashboard.html file will handle fetching and displaying the images.
    return render_template('dashboard.html')

@app.route('/get_images')
def get_images():
    """Returns a JSON list of image filenames for the JavaScript dashboard to fetch."""
    
    if not os.path.exists(LOG_FOLDER):
        return jsonify(images=[])
        
    # List all JPG files
    image_list = glob.glob(os.path.join(LOG_FOLDER, 'UNAUTHORIZED_*.jpg'))
    
    # Sort files by modification time (newest first, for display order)
    image_list.sort(key=os.path.getmtime, reverse=True)
    
    # Return just the filenames (e.g., ['UNAUTHORIZED_20251112-140000.jpg', ...])
    image_names = [os.path.basename(f) for f in image_list]
    
    return jsonify(images=image_names)

if __name__ == '__main__':
    print("Flask Server running...")
    
    # Run on local loopback IP for security since you're using one laptop
    # IMPORTANT: This server MUST be running before you start security_cam.py
    app.run(host='127.0.0.1', port=5000, debug=False)