import pathlib
import torch # For torch.cuda.is_available()
import easyocr # Added back for EasyOCR

# Import settings and utility functions from their new locations
from src.config import settings
from src.database.db_utils import init_db_connection
# Note: flush_db_batch is used within pipeline.py
# draw_text_with_background is used within pipeline.py (imported from src.utils.drawing)
# preprocess_lp_for_ocr and extract_license_plate_info_ocr are used within pipeline.py (imported from src.core.ocr_utils)

from src.core.pipeline import run_main_pipeline
from src.core.qdrant_sync import initialize_qdrant_resources # Import Qdrant initializer
# Queues (ocr_task_queue, display_results_queue) are managed within pipeline.py
# worker_stats is initialized here and passed, then managed by pipeline.py's worker
# ENABLE_LP_PREPROCESSING is managed within pipeline.py (initialized from settings, toggled by keypress there)

# Ultralytics YOLO is imported within this file's __main__ block where models are loaded.
# Other imports like cv2, numpy, time, sqlite3, datetime, threading, queue are now primarily in pipeline.py or other utils.

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).resolve().parent
    
    # Construct paths using settings
    vehicle_model_fpath = project_root / "src" / "models" / "yolo" / settings.VEHICLE_MODEL_NAME
    
    video_source_input = settings.VIDEO_SOURCE_REL
    video_source_fpath_str: str | int # Type hint for clarity
    if isinstance(video_source_input, str):
        video_source_fpath_str = str(project_root / video_source_input)
    elif isinstance(video_source_input, int):
        video_source_fpath_str = video_source_input # Use integer directly for webcam
    else:
        print(f"[ERROR] Invalid VIDEO_SOURCE_REL type in settings: {type(video_source_input)}. Must be str or int.")
        exit()
        
    lp_model_fpath_obj = project_root / settings.LP_MODEL_PATH_REL
    db_path_str = str(project_root / settings.DB_NAME)

    db_connection, db_curs = init_db_connection(db_path_str)
    
    worker_stats_shared = {'lp_detections': 0, 'ocr_successes': 0, 'tasks_processed': 0}

    ocr_reader_instance = None
    if settings.EASYOCR_LANGUAGES:
        try:
            print("[INFO] Initializing EasyOCR Reader...")
            ocr_reader_instance = easyocr.Reader(settings.EASYOCR_LANGUAGES, gpu=torch.cuda.is_available())
            print(f"[INFO] EasyOCR Reader initialized. Device: {ocr_reader_instance.device if hasattr(ocr_reader_instance, 'device') else 'CPU (default)'}")
        except Exception as e_ocr_init:
            print(f"[WARN] Failed to initialize EasyOCR Reader: {e_ocr_init}. OCR will be disabled.")
    else:
        print("[INFO] No languages specified for EasyOCR, OCR will be disabled.")

    print(f"[INFO] Starting Full Pipeline (DB: {settings.DB_NAME})...")
    print(f"[INFO] Vehicle Model: {vehicle_model_fpath}")
    print(f"[INFO] LP Model: {lp_model_fpath_obj}")
    print(f"[INFO] Video Source: {video_source_fpath_str}")
    print(f"[INFO] LP/OCR Cooldown: {settings.LP_OCR_COOLDOWN_FRAMES} frames per Track ID") 
    if settings.PRE_INFERENCE_RESIZE_WIDTH and settings.PRE_INFERENCE_RESIZE_WIDTH > 0: 
        print(f"[INFO] Pre-inference resize width for vehicle detection: {settings.PRE_INFERENCE_RESIZE_WIDTH}px") 

    from ultralytics import YOLO # YOLO needed for model loading here
    custom_lp_model = None
    vehicle_yolo_model = None
    
    try:
        device_to_use = 0 if torch.cuda.is_available() else 'cpu'
        device_str_for_to = f'cuda:{device_to_use}' if isinstance(device_to_use, int) else device_to_use
        print(f"[INFO] Attempting to load YOLO models and move to device: '{device_str_for_to}'")
        
        if not vehicle_model_fpath.exists():
            print(f"[ERROR] Vehicle model file not found: {vehicle_model_fpath}")
            if db_connection: db_connection.close()
            exit()
        vehicle_yolo_model = YOLO(vehicle_model_fpath) 
        vehicle_yolo_model.to(device_str_for_to)
        print(f"[INFO] Vehicle model loaded. Device: {vehicle_yolo_model.device if hasattr(vehicle_yolo_model, 'device') else 'Unknown'}")

        if lp_model_fpath_obj.exists():
            custom_lp_model = YOLO(lp_model_fpath_obj)
            custom_lp_model.to(device_str_for_to)
            print(f"[INFO] LP model loaded. Device: {custom_lp_model.device if hasattr(custom_lp_model, 'device') else 'Unknown'}")
        else: 
            print(f"[WARN] LP model not found: {lp_model_fpath_obj}. LP detection will be skipped.")
            
    except Exception as e: 
        print(f"[ERROR] Failed to load one or more models or move to device: {e}")
        if db_connection: db_connection.close()
        exit()
    
    # Initialize Qdrant resources before starting the pipeline
    try:
        print("[INFO] Initializing Qdrant resources...")
        initialize_qdrant_resources()
        print("[INFO] Qdrant resources initialized successfully.")
    except Exception as e_qdrant_init:
        print(f"[ERROR] Failed to initialize Qdrant resources: {e_qdrant_init}")
        print("[INFO] Proceeding without Qdrant integration for this session.")
        # Optionally, exit if Qdrant is critical:
        # if db_connection: db_connection.close()
        # exit()

    try:
        run_main_pipeline(
            video_source_path=str(video_source_fpath_str),
            vehicle_model=vehicle_yolo_model,
            lp_model_instance=custom_lp_model,
            ocr_reader_instance=ocr_reader_instance,
            db_conn=db_connection,
            db_cursor=db_curs,
            worker_stats_ref=worker_stats_shared
        )
    except Exception as e_main:
        print(f"[FATAL] Main execution error: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        if db_connection: 
            db_connection.close()
            print("[DB] Database connection closed (main).")
        print("[INFO] Application finished.")