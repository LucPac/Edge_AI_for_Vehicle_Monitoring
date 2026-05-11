# Edge AI for Vehicle Monitoring - Backend Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)
![SQLite](https://img.shields.io/badge/SQLite-3-blue.svg)
![ONNX](https://img.shields.io/badge/ONNX_Runtime-1.15-005ced.svg)

## Overview

This repository contains the backend pipeline for the Edge AI for Vehicle Monitoring system. Built with FastAPI, it serves as the central hub connecting IoT edge devices (ESP32, ESP32-CAM) with deep learning models (YOLO, CRNN) and a lightweight SQLite database to create a fully automated, real-time smart parking management system optimized for edge computing environments.

## Key Features

* **Real-time Object Detection:** Integrates a lightweight YOLO model via ONNX Runtime to detect vehicles and license plates with minimal latency.
* **Automated License Plate Recognition (ALPR):** Utilizes a CRNN-based OCR model combined with geometric image alignment algorithms for high-accuracy text extraction.
* **Hardware Synchronization:** Seamlessly communicates with ESP32 microcontrollers for RFID scanning and ESP32-CAM for continuous video streaming.
* **Edge-Optimized Storage:** Utilizes a serverless, zero-configuration SQLite database suitable for deployment on low-power devices (e.g., Raspberry Pi) without the overhead of a dedicated database server.
* **Live Dashboard Integration:** Uses WebSockets to broadcast real-time parking events, images, and OCR results to the frontend dashboard without requiring page reloads.
* **Clean Architecture:** Strictly follows the Separation of Concerns principle, dividing the system into distinct, maintainable modules (AI, Services, API, Database).

## Project Structure

```text
pipeline_test/
├── main.py                 # FastAPI application entry point, server config & DB init
├── config.py               # Global configurations & constants (Models, Hardware IPs)
├── database.py             # SQLite connection management
├── parking.db              # Auto-generated SQLite database file (created on runtime)
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
* Edge AI models (`best.onnx` and `rec_model.onnx`) placed inside the `models/` directory.
* **Recommended Editor Tool:** Install the `SQLite DB Viewer` extension (by keyshout) in Visual Studio Code to view, filter, and edit the database directly within the IDE.

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

The system uses SQLite for embedded storage. The database file (`parking.db`) and all necessary tables are automatically initialized upon starting the FastAPI server. No manual database server installation or connection strings are required.

## How to Run the Complete Pipeline

Follow these steps to start the entire vehicle monitoring pipeline:

### Step 1: Start the Backend Server

Ensure the virtual environment is activated. Run the FastAPI server and expose it to the local network so the ESP32 devices can connect:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Note: Once started, the terminal will display the local network IP address (e.g., `http://192.168.1.xxx:8000/`). Keep this terminal open.

### Step 2: Access the Web Dashboard

* **On the host machine:** Open a web browser and navigate to `http://localhost:8000/`
* **On mobile/external devices:** Ensure the device is connected to the same local network and access the IP address provided in the terminal.

### Step 3: Power Up the Edge Devices

1. **ESP32-CAM:** Power the module to initiate the live video stream. Verify the camera feed on the dashboard.
2. **ESP32 (RFID Reader):** Power the module and wait for Wi-Fi connection confirmation via the Serial Monitor.

### Step 4: System Operation

1. **Trigger:** Swipe an RFID card on the RC522 reader.
2. **Transmission:** The ESP32 submits the RFID code to the backend server via an HTTP POST request.
3. **AI Processing:** The server captures the current frame from the ESP32-CAM stream, processes it through the YOLO model for plate localization, and executes the CRNN model for optical character recognition.
4. **Real-time Update:** The system records the transaction in the SQLite database and broadcasts the event via WebSockets. The Web Dashboard instantly reflects the updated data, including the vehicle image, cropped license plate, and extracted text.

## API Reference

* `GET /video_feed`: Streams MJPEG video frames with YOLO bounding boxes.
* `POST /api/swipe`: Triggered by the ESP32 module. Captures frame, executes ALPR, updates the database, and broadcasts the event.
* `POST /api/register`: Registers a new RFID tag.
* `GET /api/logs`: Retrieves recent parking transaction history.
* `WS /ws`: WebSocket endpoint for real-time frontend synchronization.