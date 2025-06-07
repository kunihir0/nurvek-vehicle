# --- Configuration Constants ---

# File/Path Settings
VEHICLE_MODEL_NAME: str = "yolo11n.pt"
LP_MODEL_PATH_REL: str = "data/lp_yolo_dataset_for_training/train_lp_run/weights/best.pt"
VIDEO_SOURCE_REL: str = "data/input_videos/source_000.mp4" # Or use 0 for webcam
DB_NAME: str = "nurvek_detections.db"
CAMERA_ID: str = "CAM_DEV_01" # Default camera ID for this instance

# API Endpoint
API_ENDPOINT_URL: str = "http://localhost:1242/api/v1/vehicle_event"
API_LIVE_FRAME_UPDATE_URL: str = "http://localhost:1242/api/v1/internal/update_live_frame" # For pipeline to send frames
API_LIVE_FRAME_GET_URL: str = "http://localhost:1242/api/v1/live_feed_frame" # For frontend to get frames
LIVE_FRAME_SEND_INTERVAL_MAIN_LOOPS: int = 5 # Send live frame to API every N main pipeline loops
API_INTERNAL_OCR_STREAM_UPDATE_URL: str = "http://localhost:1242/api/v1/internal/ocr_stream_chunk" # For OCR worker to send stream chunks
API_INTERNAL_BACKEND_STATUS_UPDATE_URL: str = "http://localhost:1242/api/v1/internal/backend_status_update" # For pipeline to send general status updates

# Detection & OCR Parameters
VEHICLE_DETECTION_CONF_THRESHOLD: float = 0.7 # Confidence threshold for main vehicle detection
LP_OCR_COOLDOWN_FRAMES: int = 1
LP_CONFIDENCE_THRESHOLD: float = 0.65 # Min confidence for LP detection model
VALID_LP_LENGTHS: list[int] = [3, 7, 8]
EASYOCR_LANGUAGES: list[str] = ['en']
EASYOCR_ALLOWLIST: str = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
PRE_INFERENCE_RESIZE_WIDTH: int = 1280 # Resize input frame to this width for vehicle detection; 0 to disable

# Ollama Vision OCR Settings (No longer used for backend LP OCR)
# OLLAMA_API_BASE_URL: str = "http://0.0.0.0:11434/api/chat"
# VISION_OCR_MODEL_NAME: str = "gemma3:4b"
# VISION_OCR_PROMPT: str = "Extract the license plate characters from this image. Return only the alphanumeric characters. If no clear plate is visible or readable, return 'UNKNOWN_PLATE'."
# VISION_OCR_TIMEOUT_SECONDS: int = 10

# Qdrant Settings
QDRANT_HOST: str = "localhost" # Assuming Qdrant is running locally
QDRANT_PORT: int = 6333 # Default Qdrant gRPC port
QDRANT_COLLECTION_NAME: str = "nurvek_events"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # Example sentence transformer

# OCR Confirmation Logic Parameters
OCR_CONFIRMATION_ATTEMPTS: int = 5    # Number of OCR reads to collect before trying to confirm
OCR_CONFIRMATION_MIN_MATCHES: int = 2 # Min identical, valid OCR results to confirm (e.g., 2 out of 5)
CONFIRMED_LP_COOLDOWN_FRAMES: int = LP_OCR_COOLDOWN_FRAMES * 10 # Cooldown after LP is confirmed

# System Parameters
DB_BATCH_SIZE: int = 50
MAX_OCR_TASK_QUEUE_SIZE: int = 50
MAX_DISPLAY_RESULT_QUEUE_SIZE: int = 50

# Colors & Display Settings (Tuples are (B, G, R))
VEHICLE_BOX_COLOR: tuple[int, int, int] = (0, 255, 0)  # Green
LP_BOX_COLOR: tuple[int, int, int] = (0, 0, 255)       # Red
LP_TEXT_COLOR: tuple[int, int, int] = (0, 0, 255)      # Red
LP_TEXT_CONFIRMED_COLOR: tuple[int, int, int] = (0, 255, 255) # Cyan
LP_MODEL_NA_COLOR: tuple[int, int, int] = (255, 100, 100) # Light Blue/Purple
FPS_TEXT_COLOR: tuple[int, int, int] = (0, 0, 255)     # Red
TEXT_BG_COLOR: tuple[int, int, int] = (0, 0, 0)        # Black background for text
TEXT_BG_ALPHA: float = 0.5                             # Semi-transparent background

# Feature Toggles
ENABLE_LP_PREPROCESSING: bool = True # Re-enable for EasyOCR