from flask import Flask, render_template, jsonify, send_from_directory
import os
import glob

# ---------------- CONFIG ----------------
LOG_FOLDER = "alarm_logs"

# Ensure alarm_logs exists
os.makedirs(LOG_FOLDER, exist_ok=True)

# Create Flask app
# static_folder + static_url_path ensures images are served at /alarm_logs/<file>
app = Flask(__name__, static_folder=LOG_FOLDER, static_url_path='/alarm_logs')

# ---------------- ROUTES ----------------

@app.route("/")
def dashboard():
    """
    Load templates/dashboard.html
    """
    return render_template("dashboard.html")


@app.route("/get_images")
def get_images():
    """
    Returns the list of intrusion images inside alarm_logs:
    {
        "images": ["UNAUTHORIZED_20251120-152050.jpg", ...]
    }
    """
    # Find all intrusion images
    image_paths = glob.glob(os.path.join(LOG_FOLDER, "UNAUTHORIZED_*.jpg"))

    # Sort by last modified time (newest first)
    image_paths.sort(key=os.path.getmtime, reverse=True)

    image_names = [os.path.basename(p) for p in image_paths]

    return jsonify(images=image_names)


@app.route("/alarm_logs/<path:filename>")
def serve_image(filename):
    """
    Serves the actual image file from alarm_logs.
    URL example: /alarm_logs/UNAUTHORIZED_20251120-152050.jpg
    """
    return send_from_directory(LOG_FOLDER, filename)


# ---------------- MAIN ----------------

if __name__ == "__main__":
    print("🚀 Flask Server running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)