# Nurvek-Vehicle 🚗

**AI-driven vehicle surveillance and tracking system** - the first operational "arm" of Project NURVE.

This module provides real-time vehicle detection, tracking, and attribute extraction from video streams, designed for integration with the central Nurvek AI kernel.

## Features

- **Multi-stream processing** - Handle concurrent video feeds from multiple cameras
- **YOLO-based detection** - Vehicle identification with bounding boxes and confidence scores
- **License plate recognition** - OCR with upscaling for enhanced accuracy
- **Vehicle tracking** - Persistent IDs across frames with attribute extraction
- **Web dashboard** - Real-time monitoring interface (optional)

## Quick Start

### Requirements
- python3
- ollama / model: gemma3:4b (visual ocr)

### Environment Setup
```bash
uv venv .venv
source .venv/bin/activate
uv pip install .
```

### License Plate Model Training
```bash
python train_lp_detector.py
```

### Run the System

**With ROCm (AMD GPU):**
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py
```

**With CUDA (NVIDIA GPU):**
```bash
python main.py
```

### Optional: Web Dashboard
```bash
uvicorn src.api.server:app --host 0.0.0.0 --port 1242 --reload --log-level warning
```

---

*Part of the NURVE surveillance ecosystem - see `docs/plan.md` for full architecture details.*

## API Endpoints

The system exposes the following user-facing API endpoints when the server is run with `uvicorn src.api.server:app`:

- **`GET /`**: Serves the main web dashboard (`index.html`).
- **`POST /api/v1/vehicle_event`**:
  - Description: Receives vehicle event data from the processing pipeline.
  - Request Body: `VehicleEvent` Pydantic schema.
  - Response: Confirmation status.
- **`GET /api/v1/vehicle_events/log`**:
  - Description: Returns a log of all vehicle events received by the API instance.
  - Response: List of `VehicleEvent` objects.
- **`GET /api/v1/track_details/{track_id}`**:
  - Description: Returns all recorded event details for a specific `vehicle_track_id`.
  - Path Parameter: `track_id` (int).
  - Response: List of `VehicleEvent` objects matching the track ID.
- **`GET /api/v1/live_feed_frame`**:
  - Description: Returns the latest live video frame (as base64 encoded JPEG) and any current object detections on that frame.
  - Response: `LiveFrameData` object including `frame_base64` and a list of `detections`.
- **`GET /api/v1/backend_status_feed`**:
  - Description: Server-Sent Event (SSE) stream for real-time backend status updates (e.g., pipeline events, worker status).
- **`POST /api/v1/events/semantic_search`**:
  - Description: Performs a semantic search over stored vehicle events using Qdrant.
  - Request Body: `SemanticSearchQuery` object (`query_text: str`, `top_k: Optional[int]`).
  - Response: `SemanticSearchResults` object containing a list of matching event payloads and scores.
- **`GET /health`**:
  - Description: Simple health check endpoint.
  - Response: `{"status": "healthy"}`.

Internal endpoints (prefixed with `/api/v1/internal/`) are used by the backend pipeline components and are not intended for direct user interaction.