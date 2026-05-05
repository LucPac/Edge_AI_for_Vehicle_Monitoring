# Edge AI for Vehicle Monitoring - Backend Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15.0-336791.svg)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.15-005ced.svg)

## Overview

This repository contains the backend pipeline for the **Edge AI for Vehicle Monitoring** system. Built with FastAPI, it serves as the central hub connecting IoT edge devices (ESP32, ESP32-CAM) with deep learning models (YOLO, CRNN) and a PostgreSQL database to create a fully automated, real-time smart parking management system.

## Key Features

* **Real-time Object Detection:** Integrates a lightweight YOLO model via ONNX Runtime to detect vehicles and license plates with minimal latency.
* **Automated License Plate Recognition (ALPR):** Utilizes a CRNN-based OCR model combined with geometric image alignment algorithms for high-accuracy text extraction.
* **Hardware Synchronization:** Seamlessly communicates with ESP32 microcontrollers for RFID scanning and ESP32-CAM for continuous video streaming.
* **Live Dashboard Integration:** Uses WebSockets to broadcast real-time parking events, images, and OCR results to the frontend dashboard without requiring page reloads.
* **Clean Architecture:** Strictly follows the *Separation of Concerns* principle, dividing the system into distinct, maintainable modules (AI, Services, API, Database).

## Project Structure

```text
pipeline_test/
├── main.py                 # FastAPI application entry point & server config
├── config.py               # Global configurations & constants (DB, Models, IP)
├── database.py             # PostgreSQL connection pool
├── models.py               # Pydantic data validation models
├── requirements.txt        # Python dependencies
├── ai/                     # Artificial Intelligence Module
│   ├── detection.py        # YOLO object detection wrapper
│   ├── recognition.py      # CRNN OCR processing
│   └── image_processing.py # Image alignment, cropping, and filtering
├── services/               # Core Business Logic & Background Tasks
│   ├── camera.py           # Background video streaming & frame caching
│   ├── vehicle.py          # ALPR logic triggered by RFID events
│   └── websocket.py        # Real-time WebSocket connection manager
├── api/                    # Application Programming Interfaces
│   ├── routes.py           # RESTful endpoints (Swipe, Register, Logs)
│   └── video.py            # MJPEG video streaming endpoints
└── static/                 # Frontend assets and generated media
    ├── images/             # Full captured frames
    ├── crops/              # Cropped license plate images
    └── index.html          # Web Dashboard
```

## Installation & Setup

### 1. Prerequisites

* Python 3.10 or higher.
* PostgreSQL installed and running.
* Edge AI models (`best.onnx` and `rec_model.onnx`) placed inside the `models/` directory.

### 2. Environment Setup

Clone the repository and navigate to the project directory:

```bash
git clone [https://github.com/LucPac/Edge_AI_for_Vehicle_Monitoring.git](https://github.com/LucPac/Edge_AI_for_Vehicle_Monitoring.git)
cd Edge_AI_for_Vehicle_Monitoring/pipeline_test
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Database Configuration

Update the PostgreSQL credentials in `config.py` to match your local database setup:

```python
DB_CONFIG = {
    "dbname": "P_Dashboard",
    "user": "postgres",
    "password": "your_password", 
    "host": "localhost",
    "port": "5432"
}
```

## How to Run the Complete Pipeline

Follow these steps to start the entire vehicle monitoring pipeline (Server + Web + Hardware):

### Step 1: Start the Backend Server

Ensure your virtual environment is activated. Run the FastAPI server and expose it to your local network so the ESP32 devices can connect:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

*Note: Once started, the terminal will display your local network IP address (e.g., `http://192.168.1.xxx:8000/`). Keep this terminal open.*

### Step 2: Access the Web Dashboard

* **On your main laptop:** Open your browser and go to `http://localhost:8000/`
* **On your mobile/other devices:** Ensure you are connected to the same Wi-Fi network and access the IP address shown in your terminal.

### Step 3: Power Up the Edge Devices

1. **ESP32-CAM:** Power it on to start the live video stream. Check the dashboard to verify that the camera feed is running smoothly.
2. **ESP32 (RFID Reader):** Power it on and wait for it to connect to the Wi-Fi. The Serial Monitor will display `[+] Da ket noi WiFi!` when ready.

### Step 4: Test the Pipeline

1. **Trigger:** Swipe an RFID card on the RC522 reader.
2. **Transmission:** The ESP32 sends the RFID code to the backend server via a POST request.
3. **AI Processing:** The server immediately captures a frame from the ESP32-CAM stream, passes it through the YOLO model for plate detection, and uses the CRNN model for OCR recognition.
4. **Real-time Update:** The server saves the data to PostgreSQL and broadcasts the event. The Web Dashboard instantly updates with the vehicle's image, the cropped license plate, and the extracted text via WebSockets.

## API Reference

* `GET /video_feed`: Streams MJPEG video frames with YOLO bounding boxes.
* `POST /api/swipe`: Triggered by ESP32 via RFID swipe. Captures frame, runs ALPR, updates DB, and broadcasts event via WS.
* `POST /api/register`: Registers a new RFID tag and links it to a user and vehicle plate.
* `GET /api/logs`: Retrieves the recent parking history for the dashboard.
* `WS /ws`: WebSocket endpoint for real-time frontend updates.