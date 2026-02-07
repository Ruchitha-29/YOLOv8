🚀 AI-Based Hybrid Surveillance System

Smart. Efficient. Real-world ready.
An AI-powered hybrid surveillance system that combines IoT sensors and computer vision to deliver accurate, real-time intrusion detection while minimizing false alarms.

Problem Statement
Traditional surveillance systems either rely solely on sensors (high false positives) or only computer vision (high computational cost). This project bridges the gap by intelligently combining both approaches to solve a real-world security challenge efficiently.

💡 Solution Overview

This system follows an event-driven hybrid architecture:

A PIR motion sensor detects movement

Triggers a YOLOv8-based AI model for verification

Confirms human presence using object detection & face recognition

Activates alerts only when a real threat is verified

✅ Reduced false alarms
✅ Optimized compute usage
✅ Faster response time

🧠 Key Features

🔔 Trigger-and-Verify Mechanism using PIR + AI
🤖 YOLOv8 Human Detection
👤 Face Recognition for identity verification with custom dataset 
⚡ Real-time intrusion alerts
🌍 Designed for real-world surveillance scenarios (border security, restricted zones, smart campuses)

Tech Stack:
Hardware
-Arduino Uno
-PIR Motion Sensor
-USB Webcam
-Buzzer & LED

Software
-Python 3
-YOLOv8 (Ultralytics)
-OpenCV
-PyTorch
-PySerial

⚙️ How It Works
PIR sensor detects motion

Arduino sends trigger signal to host system

Python application captures video frame

YOLOv8 verifies human presence

Face recognition confirms identity

Alarm/dashboard alert is activated if intrusion is confirmed

📊 Results & Impact
🚫 Significant reduction in false positives
⚡ Faster intrusion verification
💰 Cost-effective compared to always-on AI systems
🔐 Reliable for security-critical environments

