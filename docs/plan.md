# Project Nurvek: Comprehensive Development Plan

**Version:** 1.0  
**Date:** June 2, 2025  
**Project Lead (Conceptual):** User & Gemini  
**Focus:** Establishing a dystopian AI surveillance system ("Nurvek") with an initial Proof of Concept (PoC) for its "Nurvek-Vehicle" tracking arm, using GTA V / FiveM as a simulated real-world environment to demonstrate core functionalities and serve as a development base for potential future real-world applications.

## 1. Project NURVE: Overall Vision & Concept

### 1.1. Core Idea & Name

**Project Name:** NURVE

**Concept:** An AI-driven, potentially omnipresent surveillance and "security optimization" system. The central AI kernel is referred to as "Nurvek."

**Architectural Philosophy:** Nurvek acts as a central intelligence, receiving and processing data from various specialized "arms" or modules. Each arm is responsible for a specific domain of surveillance or data collection.

### 1.2. Dystopian Setting

**Inspiration:** Cyberpunk 2077 aesthetics and themes – megacorporations, advanced technology, social stratification, constant surveillance.

**PoC Environment:** A GTA V FiveM server, modified to simulate a dystopian city. This allows for controlled testing of real-time video feed processing from various sources (mock drones, fixed traffic cameras, pedestrian/vehicle phone feeds). It is important to note that this FiveM environment serves as a robust platform for demonstrating the system's capabilities and iterating on its design, with the foundational work intended to be adaptable for future development towards real-world applications.

**Narrative Context (for FiveM):** The city-state operates under the pervasive strategic guidance of Nurvek, an advanced AI system that has become integral to governance. Nurvek's directives shape policy, resource allocation, and security protocols, effectively making it the operational core of the state. While human administrative structures may still exist, their autonomy is increasingly nominal, with Nurvek's 'optimizations' and 'security mandates' holding decisive sway. The populace experiences this as an omnipresent, efficient, yet often opaque authority.

### 1.3. Conceptual Core System Components of NURVE

**Sensor Layer (Data Sources for Arms):**
- **PoC:** In-game GTA V FiveM video streams (drone, traffic cams, phone POVs) provided via URLs.
- **Conceptual Future:** Real-world CCTV, drones, IoT devices, biometric scanners, digital footprints.

**Ingestion & Processing Pipeline (within each "Arm"):**
- Specialized AI models for domain-specific detection and analysis.
- Data structuring and pre-processing for the Nurvek kernel.

**AI Core / Kernel ("Nurvek"):**
- The Nurvek Kernel is the central intelligence of the AI-driven state. It not only processes data from all operational arms but also formulates strategic objectives, issues directives for 'civic optimization' and 'threat neutralization,' and autonomously manages many aspects of urban function. Its decisions are the de facto policies of the state.

**Control Layer & Output:**
- Interfaces for State Enforcers and System Custodians, providing curated data streams and actionable directives from Nurvek.
- Alerting systems.
- APIs for triggering interventions (e.g., dispatch of state resources, area lockdowns, information dissemination as per Nurvek's protocols).
- Comprehensive data archives, forming the immutable 'State Record' as defined and maintained by Nurvek, underpinning all AI-driven governance.

### 1.4. Modular "Arm" Architecture

- **Nurvek (Kernel):** The central brain, the core of the AI-driven state governance.
- **Nurvek-Vehicle (Arm):** Focus of this initial PoC – responsible for vehicle detection, tracking, and attribute extraction.
- **Future Arms (Conceptual):** Nurvek-Pedestrian, Nurvek-Infrastructure, Nurvek-Comms, etc.

## 2. Nurvek-Vehicle Arm: Detailed Plan

This is the first "arm" to be developed, focusing on vehicle-related surveillance.

### 2.1. Purpose

To serve as a primary mobile asset surveillance component for the Nurvek State AI. It will detect, track, identify, and extract detailed attributes of all vehicular assets within its operational zones, feeding this intelligence directly into the Nurvek kernel to inform its strategic city management and enforcement protocols.

### 2.2. Data Ingestion (PoC in FiveM)

- **Source:** Live video streams from the FiveM server (e.g., from a drone plugin, fixed camera scripts) provided as URLs.
- **Multi-Stream Processing:** The system should be designed to conceptually handle multiple concurrent video streams. Each stream will be associated with a unique camera_id (e.g., `CAM_DOWNTOWN_SQ_01`, `DRONE_PATROL_ALPHA_FEED`).

### 2.3. Core Vehicle Perception Pipeline (Workflow per stream)

1. **Frame Acquisition:** Read frame from the specific camera URL.
2. **Preprocessing:** Resize, normalize frame for the YOLO model input requirements.
3. **YOLO Inference (Vehicle Detection & Basic Classification):**
   - Utilize the provided `yolo11l.pt` model (or other suitable Ultralytics YOLO models like YOLOv8 series).
   - Detect primary objects: cars, motorcycles, trucks, bicycles. Pedestrians can also be detected for broader context.
   - Output: Bounding boxes, class labels (e.g., "car"), confidence scores.
4. **Vehicle Tracking (Persistent IDs per camera):**
   - Employ Ultralytics YOLO's track mode.
   - Assign a unique `vehicle_track_id` (e.g., `CAM_ID_VEH_XYZ`) that persists for a vehicle as long as it's reliably tracked by that specific camera.
5. **Detailed Attribute Extraction Modules (for each tracked vehicle):**

   **A. License Plate Module:**
   - Crop vehicle region based on YOLO detection.
   - Run a specialized license plate detector model on the cropped region.
   - (Optional) Implement/integrate an AI Upscaler (e.g., Real-ESRGAN). This should be triggered if an initial OCR attempt on the cropped and detected license plate region fails or yields a low confidence score. After upscaling, OCR will be attempted again. Proactive triggering based on initial plate region resolution (if too small) can also be considered.
   - Perform OCR on the (potentially upscaled) plate image using Tesseract or a specialized OCR model.
   - Output: `license_plate_text`, `ocr_confidence`, `plate_bbox_relative_to_car`.

   **B. Color Classification Module:**
   - Crop vehicle region.
   - Run a color classifier (e.g., a small CNN or a heuristic algorithm analyzing dominant colors).
   - Output: `vehicle_color` (e.g., "blue", "dark_red"), `color_confidence`.

   **C. Damage Assessment Module (Conceptual for PoC, iterative development) (OPTIONAL):**
   - Initial PoC: Could be a placeholder or based on very obvious visual cues if time permits.
   - Future: Train a model to recognize common GTA damage types (dents, broken windows, smoke).
   - Output: `damage_state` (e.g., "none", "minor_front_dent", "smoking"), `damage_confidence`.

   **D. Vehicle Model/Type Refinement (Conceptual for PoC, iterative development):**
   - If base YOLO provides "car," a secondary classifier could refine this to GTA-specific models (e.g., "Buffalo," "Banshee") or broader types ("sedan," "SUV").
   - Requires a custom classifier trained on images of GTA vehicle models.
   - Output: `vehicle_model_type`, `model_confidence`.

### 2.4. Contextual Memory (Vector Database)

**Technology Choice (Preferred):** Qdrant (or similar purpose-built vector database).

**Purpose:** To store and rapidly query vector embeddings representing detected entities (initially focused on vehicles, potentially faces later if that module is developed and ethically approved). This enables:
- Re-identification of the same vehicle across different sightings by the same camera more robustly, or potentially by Nurvek across different cameras.
- Storing historical data and associations.

**Functionality (for Nurvek-Vehicle):**
- Store embeddings of unique vehicle characteristics (visual signature beyond just model/plate).
- Link `vehicle_track_ids` to `license_plate_text` and other persistent attributes.
- Enable Nurvek to query for vehicles based on similarity or known features.
- Facilitate building associations: vehicle to common locations, vehicle to other frequently co-occurring vehicles (if Nurvek handles this higher-level logic).

### 2.5. Structured Data Output from Nurvek-Vehicle Arm

The arm will output structured data (e.g., JSON) for each significant update on a tracked vehicle.

```json
{
  "timestamp_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "camera_id": "CAM_SECTOR_A_LIGHTPOST_01",
  "vehicle_track_id": "CAM_A_01_VEH_101", // Unique ID from this camera's tracker
  "base_class": "car", // e.g., car, motorcycle, truck
  "confidence_base_class": 0.92,
  "bounding_box_frame_coords": [x, y, w, h], // In current frame
  "world_coordinates_estimate_gta": [gta_x, gta_y, gta_z], // If mappable from FiveM
  "attributes": {
    "license_plate": {
      "text": "NURVEK7", // Example
      "ocr_confidence": 0.88,
      "detected": true // boolean
    },
    "color": {
      "primary_color": "blue",
      "confidence": 0.95
    },
    "damage_state": { // Conceptual
      "description": "minor_dent_driver_door",
      "level": "minor", // none, minor, moderate, severe
      "confidence": 0.70
    },
    "model_type": { // Conceptual
      "name": "Buffalo STX", // Example GTA model name
      "confidence": 0.80
    }
  },
  "current_speed_estimate_kph": 60, // If derivable from frame changes
  "heading_degrees": 90, // If derivable
  "flags": ["entering_restricted_zone"] // Dynamic flags Nurvek might request or the arm infers
}
```

### 2.6. Communication with Central Nurvek Kernel

- **Data Streaming:** The Nurvek-Vehicle arm will stream these structured objects to a designated endpoint/service that Nurvek consumes (e.g., via a message queue like Kafka/RabbitMQ, or a direct API).
- **Event-Driven Alerts (Conceptual):** For high-priority events (e.g., watchlist license plate detected), the arm might push an immediate alert.
- **API for Nurvek (Conceptual):** The arm (or an intermediary data store it populates) could expose an API for Nurvek to query specific information (e.g., last known location of a plate, vehicles currently in a specific zone).

### 2.7. Intended Use / Impact (Simulated in FiveM Dystopia)

- Sensor data directly populates Nurvek's strategic overview, enabling real-time situational awareness for State Enforcers.
- Automated flagging against Nurvek's dynamic 'Persons/Vehicles of Interest' matrix and behavioral anomaly parameters.
- Critical input for Nurvek's predictive modeling, resource deployment algorithms, and automated directive generation.
- Forms a core part of the State Record, ensuring all vehicular movements and related incidents are logged for Nurvek's continuous analysis and governance optimization.

## 3. Proof of Concept (PoC) - Phase 0: Initial Setup & Environment

**Goal:** Establish a clean, reproducible Python development environment using uv, set up a logical project directory structure, and prepare for integrating the yolo11l.pt model.

### 3.1. Project Directory Structure

**User Preference:** No `__init__.py` files will be used, relying on Python 3.3+ namespace package capabilities.

**User to Provide `<main_package_name>`:** The user will specify the name for the main package directory within `src/`. For this plan, we will use the placeholder `<main_package_name>`.

```
nurvek-vehicle/
├── .git/                     # Git directory (assumed to exist)
├── .venv/                    # Virtual environment (will be created by uv)
├── src/                      # Source code
│   ├── core/             # Core logic (video processing, detection)
│   │   └── detection.py  # Script for detection logic
│   ├──  models/           # Model-related files
│   │   └── yolo/
│   │       └── yolo11l.pt # User's provided model
│   ├── streams/          # For handling video stream inputs
│   ├── utils/            # Utility functions
│   │   └── main.py           # Main entry point for running/testing
│   ├── utils/              # Helper scripts
│   └── tests/                # Unit and integration tests
├── data/                     # For local test video files, output, etc.
│   └── input_videos/         # Place test video files here
│   └── output_frames/        # For saving processed frames (optional)
├── notebooks/                # Jupyter notebooks for experimentation (optional)
├── pyproject.toml            # Project metadata and dependencies (for uv)
├── README.md                 # Project overview
└── .gitignore                # Files to ignore in Git
```

**Action for User:**
1. Create the main `nurvek-vehicle` project directory.
2. Inside `nurvek-vehicle`, create the directories as per the structure above.
3. Place the `yolo11l.pt` file into `src/models/yolo/`.

### 3.2. uv Environment Setup

1. **Install uv:** If not already installed system-wide, follow official Astral installation instructions (e.g., `curl -LsSf https://astral.sh/uv/install.sh | sh` for macOS/Linux). Verify with `uv --version`.

2. **Create & Activate Virtual Environment:**
   - Navigate to the project root (`nurvek-vehicle/`).
   - Create venv: `uv venv` (this creates `.venv/`)
   - Activate venv:
     - macOS/Linux: `source .venv/bin/activate`
     - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
     - Windows (CMD): `.venv\Scripts\activate.bat`

### 3.3. Initial Dependencies (pyproject.toml)

Create `pyproject.toml` in the project root (`nurvek-vehicle/`) with the following content:

```toml
[project]
name = "nurvek-vehicle-project" # Overall project name (can differ from <main_package_name>)
version = "0.1.0"
description = "Vehicle tracking arm PoC for the Nurvek AI system."
requires-python = ">=3.9" # Or your preferred Python 3 version
dependencies = [
    "ultralytics",    # For YOLO models
    "opencv-python"   # For video processing and display
    # numpy and torch will be pulled in by ultralytics
]

[project.optional-dependencies]
dev = [
    # "ruff", # Example: for linting/formatting if desired
    # "pytest", # Example: for testing if desired
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

**Install Dependencies:**
- Ensure the uv virtual environment is active.
- From the project root, run: `uv pip install .`
  (This command installs the dependencies listed in pyproject.toml into your active virtual environment.)

### 3.4. Verification

1. **Check Installed Packages:** In the active venv, run `uv pip list`. Verify ultralytics, opencv-python, torch, numpy, etc., are present.

2. **Basic Python Import Test:** Create a temporary Python script or use the interpreter:

```python
import cv2
from ultralytics import YOLO # Preferred way to check ultralytics
import torch

print(f"OpenCV version: {cv2.__version__}")
print("Ultralytics YOLO imported successfully.")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}") # Important if GPU is intended
```

If this runs without import errors, the environment is likely set up correctly.

## 4. Proof of Concept (PoC) - Phase 1: First Video Detection

**Goal:** Load a video (initially a local file, then a FiveM URL), run the yolo11l.pt model for object detection, and display the results with bounding boxes.

### 4.1. Prerequisites

- Phase 0 (Setup & Environment) completed successfully.
- A sample video file (e.g., .mp4 of street traffic) placed in `data/input_videos/`.

### 4.2. Steps

1. **Create Detection Script:**
   Create a Python file, e.g., `src/main.py` or `src/core/detection_poc.py`.

2. **Implement Script Logic:**

```python
import cv2
from ultralytics import YOLO
import pathlib # To construct model path robustly

def run_detection_on_video(video_source, model_path_str):
    """
    Runs YOLO object detection on a video source and displays the results.

    Args:
        video_source (str or int): Path to video file, URL, or camera index (0 for webcam).
        model_path_str (str): Path to the YOLO model file (.pt).
    """
    model_path = pathlib.Path(model_path_str)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    print(f"Successfully opened video source: {video_source}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or error.")
            break

        # Perform inference
        # For PoC, detect all COCO classes or specify relevant ones like cars/persons
        # results = model.predict(frame, verbose=False) # verbose=False to reduce console spam
        # For tracking:
        results = model.track(frame, persist=True, verbose=False)

        if results and results[0].boxes is not None:
            annotated_frame = results[0].plot() # plot() draws bboxes and labels
        else:
            annotated_frame = frame # Show original frame if no detections

        cv2.imshow("Nurvek-Vehicle PoC - Detection Feed", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Released video capture and destroyed windows.")

if __name__ == "__main__":
    # --- Configuration for PoC ---
    # Construct the model path relative to this script or use an absolute path
    # Assuming this script is in src/<main_package_name>/
    
    import os
    from pathlib import Path
    
    # Get the current script directory
    script_dir = Path(__file__).parent
    
    # Construct path to model file
    model_path = script_dir / "models" / "yolo" / "yolo11l.pt"
    
    # Example video source - adjust as needed
    video_source = "../../data/input_videos/sample_video.mp4"  # Local video file
    # video_source = 0  # For webcam
    # video_source = "http://your-fivem-server.com/stream"  # For FiveM stream
    
    print("Starting Nurvek-Vehicle Detection PoC...")
    print(f"Model path: {model_path}")
    print(f"Video source: {video_source}")
    
    run_detection_on_video(video_source, str(model_path))
```

### 4.3. Testing

1. **Prepare Test Video:** Place a sample video file in `data/input_videos/sample_video.mp4`
2. **Run the Script:** From the project root with the virtual environment activated:
   ```bash
   python ./nurvek-v.py
   ```
3. **Expected Output:** A window should open showing the video with bounding boxes around detected objects
4. **Quit:** Press 'q' to exit the detection loop

### 4.4. Next Steps

- Test with different video sources (webcam, URL streams)
- Implement vehicle-specific filtering
- Add license plate detection module
- Integrate with vector database for tracking persistence
