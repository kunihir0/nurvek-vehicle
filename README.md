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