import cv2
import numpy as np
import time
import pathlib # Keep for potential future use, though settings might handle paths
import torch
import sqlite3 # Keep for type hints if db_conn/cursor are passed around
import datetime
import easyocr # Added back for EasyOCR
import threading
import queue
import requests # For making HTTP requests to the API
import base64 # For encoding images
import io # For in-memory bytes buffer
from typing import Any, Optional, List, Dict, Tuple # Import necessary types

from src.config import settings
from src.core.schemas import VehicleEvent, VehicleAttributes, LicensePlateData # Import Pydantic schemas
from src.database.db_utils import flush_db_batch # init_db_connection will be called in main.py
from src.utils.drawing import draw_text_with_background
from src.core.ocr_utils import extract_license_plate_info_ocr
from src.core.qdrant_sync import embed_and_store_event # Qdrant integration

# --- Module-level globals for queues and stats ---
ocr_task_queue = queue.Queue(maxsize=settings.MAX_OCR_TASK_QUEUE_SIZE)
display_results_queue = queue.Queue(maxsize=settings.MAX_DISPLAY_RESULT_QUEUE_SIZE)
# worker_stats will be passed as a mutable dictionary (worker_stats_ref)

# Module-level global for the preprocessing toggle
ENABLE_LP_PREPROCESSING = settings.ENABLE_LP_PREPROCESSING # Initialize from settings

def _post_backend_status(source: str, event_type: str, data: Dict[str, Any]):
    """Helper function to post backend status updates."""
    try:
        payload = {
            "source": source,
            "event_type": event_type,
            "data": data,
            # timestamp is added by Pydantic model in API server
        }
        requests.post(settings.API_INTERNAL_BACKEND_STATUS_UPDATE_URL, json=payload, timeout=0.5)
    except Exception as e:
        # print(f"[PIPELINE_STATUS_POST_ERROR] Failed to post backend status: {e}") # Can be too verbose
        pass # Best effort, don't let status posting crash pipeline

def ocr_worker_function(
    lp_model: Any,  # YOLO model instance
    ocr_reader_instance: Optional[easyocr.Reader], # Added EasyOCR reader
    db_conn: sqlite3.Connection,
    db_cursor: sqlite3.Cursor,
    worker_stats_dict: dict
) -> None:
    """Processes OCR tasks from a queue, including LP detection and OCR using EasyOCR."""
    print("[WORKER_OCR] OCR Worker thread started.")
    pending_db_records_worker: List[Tuple[Any, ...]] = []
    while True:
        try:
            _post_backend_status("ocr_worker", "status", {"state": "idle", "queue_size": ocr_task_queue.qsize()})
            task_data = ocr_task_queue.get(timeout=1)
            if task_data is None:
                print("[WORKER_OCR] Sentinel received, flushing DB and exiting.")
                _post_backend_status("ocr_worker", "status", {"state": "shutting_down"})
                flush_db_batch(db_conn, db_cursor, pending_db_records_worker)
                break
            
            _post_backend_status("ocr_worker", "task_received", {"queue_size": ocr_task_queue.qsize()})
            worker_stats_dict['tasks_processed'] += 1
            vehicle_crop, db_record_base_info, track_id_for_worker, frame_num_for_worker = task_data
            
            _post_backend_status("ocr_worker", "processing_start", {"track_id": track_id_for_worker, "frame_num": frame_num_for_worker})
            # Pass the current state of ENABLE_LP_PREPROCESSING from this module

            lp_detected_flag, ocr_success_flag, ocr_text_result, lp_conf_result, lp_bbox_result, raw_ocr_text_for_console, final_lp_image_for_ocr = \
                extract_license_plate_info_ocr(vehicle_crop, lp_model, ocr_reader_instance)

            lp_image_base64_worker: Optional[str] = None
            if final_lp_image_for_ocr is not None and final_lp_image_for_ocr.size > 0:
                try:
                    is_success, buffer = cv2.imencode(".jpg", final_lp_image_for_ocr)
                    if is_success:
                        lp_image_base64_worker = base64.b64encode(buffer).decode('utf-8')
                except Exception as e_encode:
                    print(f"[WORKER_OCR_ERROR] Failed to encode LP image: {e_encode}")

            if lp_detected_flag: worker_stats_dict['lp_detections'] += 1
            if ocr_success_flag: worker_stats_dict['ocr_successes'] += 1

            # lp_conf_display_str = f"{lp_conf_result:.2f}" if lp_conf_result is not None else "0.00" # Commented out as per user request to reduce spam
            # if raw_ocr_text_for_console: 
            #      print(f"[OCR_WORKER_RAW_TEXT] TrackID: {track_id_for_worker}, Frame: {frame_num_for_worker}, LP_Conf: {lp_conf_display_str}, RAW_OCR: '{raw_ocr_text_for_console}' (Valid: {ocr_success_flag})")
            # elif lp_detected_flag: 
            #      print(f"[OCR_WORKER_RAW_TEXT] TrackID: {track_id_for_worker}, Frame: {frame_num_for_worker}, LP_Conf: {lp_conf_display_str}, RAW_OCR: <No text from EasyOCR>")

            db_record_final_list = list(db_record_base_info.values())
            if lp_detected_flag:
                db_record_final_list[6] = round(lp_conf_result, 4) if lp_conf_result else None
                if lp_bbox_result:
                    db_record_final_list[7] = f"{lp_bbox_result[0]},{lp_bbox_result[1]},{lp_bbox_result[2]},{lp_bbox_result[3]}"
                db_record_final_list[8] = ocr_text_result 
            
            pending_db_records_worker.append(tuple(db_record_final_list))
            if len(pending_db_records_worker) >= settings.DB_BATCH_SIZE: 
                flush_db_batch(db_conn, db_cursor, pending_db_records_worker)

            display_info_for_main = {
                'track_id': track_id_for_worker,
                'frame_num': frame_num_for_worker,
                'lp_detected_flag': lp_detected_flag,
                'ocr_success_flag': ocr_success_flag,
                'ocr_text_result': ocr_text_result,
                'lp_conf_result': lp_conf_result,
                'lp_bbox_result': lp_bbox_result,
                'lp_image_base64': lp_image_base64_worker 
            }
            try:
                display_results_queue.put_nowait(display_info_for_main)
            except queue.Full:
                pass 
        except queue.Empty:
            # _post_backend_status("ocr_worker", "status", {"state": "idle_timeout", "queue_size": ocr_task_queue.qsize()}) # Can be too verbose
            continue
        except Exception as e:
            print(f"[WORKER_OCR_ERROR] Error in ocr_worker_function: {e}")
            _post_backend_status("ocr_worker", "error", {"error_message": str(e)})
    _post_backend_status("ocr_worker", "status", {"state": "stopped"})
    print(f"[WORKER_OCR] OCR Worker thread stopped. Processed {worker_stats_dict['tasks_processed']} tasks. LP Detections: {worker_stats_dict['lp_detections']}, OCR Successes: {worker_stats_dict['ocr_successes']}.")


def run_main_pipeline( 
    video_source_path: str, 
    vehicle_model: Any, 
    lp_model_instance: Optional[Any],
    ocr_reader_instance: Optional[easyocr.Reader], # Added EasyOCR reader
    db_conn: sqlite3.Connection,
    db_cursor: sqlite3.Cursor,
    worker_stats_ref: dict
) -> None:
    global ENABLE_LP_PREPROCESSING
    gui_works = True
    window_name = "Nurvek PoC - Modular Pipeline"
    
    if hasattr(vehicle_model, 'device'): print(f"[INFO] Vehicle model device: {vehicle_model.device}")
    elif hasattr(vehicle_model, 'model') and hasattr(vehicle_model.model, 'device'): print(f"[INFO] Vehicle model device: {vehicle_model.model.device}")

    cap = cv2.VideoCapture(video_source_path)
    if not cap.isOpened(): 
        print(f"[ERROR] Could not open video: {video_source_path}")
        return
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video source opened: {video_source_path} ({orig_w}x{orig_h})")

    vehicle_names = vehicle_model.model.names if hasattr(vehicle_model, 'model') else vehicle_model.names
    # Initialize all known classes to be active by default
    active_filters: Dict[str, bool] = {class_name: True for class_name in vehicle_names.values()}
    
    filter_keys_map: Dict[int, str] = {}
    preferred_vehicle_classes: List[str] = ['car','motorcycle','truck','bicycle','bus']
    assignable_keys: List[int] = [ord(str(i)) for i in range(1,10)]
    temp_assigned_classes: set[str] = set()
    current_key_idx: int = 0

    for p_cls in preferred_vehicle_classes:
        if p_cls in vehicle_names.values() and current_key_idx < len(assignable_keys):
            # active_filters[p_cls] = True # Already set to True by default initialization
            filter_keys_map[assignable_keys[current_key_idx]] = p_cls
            temp_assigned_classes.add(p_cls)
            current_key_idx += 1
    if current_key_idx < len(assignable_keys):
        other_classes = sorted(list(set(vehicle_names.values()) - temp_assigned_classes))
        for o_cls in other_classes:
            if current_key_idx < len(assignable_keys):
                # active_filters[o_cls] = True # Already set to True by default initialization
                filter_keys_map[assignable_keys[current_key_idx]] = o_cls
                current_key_idx +=1
            else: break
            
    display_ocr_on_gui: bool = True 
    print("\n--- Controls ---") 
    if filter_keys_map:
        for k_code, cls_item in filter_keys_map.items(): print(f"Press '{chr(k_code)}' to toggle filter: {cls_item}")
    else: print("No filterable classes for keys 1-9.")
    print(f"Press 'o' to toggle OCR text display (Currently: {'ON' if display_ocr_on_gui else 'OFF'})")
    print(f"Press 'p' to toggle LP Preprocessing (Currently: {'ON' if ENABLE_LP_PREPROCESSING else 'OFF'})") 
    print("Press 'q' to quit."); print("----------------\n")

    frames_processed: int = 0
    total_loop_time: float = 0.0
    total_vehicles_detected_main: int = 0
    total_lp_ocr_tasks_queued: int = 0
    ocr_queue_full_count: int = 0
    unique_vehicle_track_ids: set[int] = set()
    confirmed_lp_texts_set: set[str] = set()
    start_time_overall: float = time.time()

    # State for live frame API sending
    live_frame_api_available: bool = True
    live_frame_connection_failures: int = 0
    MAX_LIVE_FRAME_FAILURES: int = 3 # Max retries before disabling
    
    tracked_vehicle_data: Dict[int, Dict[str, Any]] = {}

    ocr_thread = threading.Thread(
        target=ocr_worker_function,
        args=(lp_model_instance, ocr_reader_instance, db_conn, db_cursor, worker_stats_ref),
        daemon=True
    )
    ocr_thread.start()

    try:
        while cap.isOpened():
            loop_start_time = time.time()
            ret, original_frame = cap.read()
            if not ret: 
                print("[INFO] Video stream ended.")
                break
            frames_processed += 1
            _post_backend_status("pipeline_main_loop", "frame_processing_start", {"frame_num": frames_processed})
            
            inference_frame, s_x, s_y = original_frame, 1.0, 1.0
            if settings.PRE_INFERENCE_RESIZE_WIDTH and settings.PRE_INFERENCE_RESIZE_WIDTH > 0 and orig_w > settings.PRE_INFERENCE_RESIZE_WIDTH: 
                s_x = settings.PRE_INFERENCE_RESIZE_WIDTH / orig_w
                new_h = int(orig_h * s_x)
                inference_frame = cv2.resize(original_frame, (settings.PRE_INFERENCE_RESIZE_WIDTH, new_h), interpolation=cv2.INTER_AREA)
                s_y = new_h / orig_h 
                if frames_processed == 1: print(f"[INFO] Resizing for vehicle detection to width: {settings.PRE_INFERENCE_RESIZE_WIDTH}")
            
            display_image = original_frame.copy() 
            live_feed_frame = original_frame.copy() 

            _post_backend_status("pipeline_main_loop", "vehicle_detection_start", {"frame_num": frames_processed})
            vehicle_results = vehicle_model.track(inference_frame, persist=True, verbose=False, half=True, conf=settings.VEHICLE_DETECTION_CONF_THRESHOLD)
            _post_backend_status("pipeline_main_loop", "vehicle_detection_end", {"frame_num": frames_processed, "num_results": len(vehicle_results[0].boxes) if vehicle_results and vehicle_results[0].boxes else 0})
            
            current_frame_detections_for_api: List[Dict[str, Any]] = []

            try:
                while not display_results_queue.empty():
                    ocr_attempt_result = display_results_queue.get_nowait()
                    track_id_from_worker = ocr_attempt_result.get('track_id')

                    if track_id_from_worker is not None and track_id_from_worker in tracked_vehicle_data:
                        track_info = tracked_vehicle_data[track_id_from_worker]
                        
                        text_attempt = ocr_attempt_result.get('ocr_text_result')
                        conf_attempt = ocr_attempt_result.get('lp_conf_result')
                        bbox_attempt = ocr_attempt_result.get('lp_bbox_result')
                        lp_detected_flag = ocr_attempt_result.get('lp_detected_flag')
                        ocr_success_flag = ocr_attempt_result.get('ocr_success_flag')

                        lp_image_b64_from_worker = ocr_attempt_result.get('lp_image_base64')
                        if lp_image_b64_from_worker:
                            track_info['lp_image_base64'] = lp_image_b64_from_worker

                        if lp_detected_flag:
                            if ocr_success_flag and text_attempt:
                                attempt_data_to_store = {
                                    "text": text_attempt,
                                    "confidence": conf_attempt,
                                    "bbox": bbox_attempt
                                }
                                track_info['lp_ocr_attempts'].append(attempt_data_to_store)
                                track_info['lp_ocr_attempts'] = track_info['lp_ocr_attempts'][-settings.OCR_CONFIRMATION_ATTEMPTS:]
                                
                                if not track_info['lp_confirmed'] and len(track_info['lp_ocr_attempts']) >= settings.OCR_CONFIRMATION_MIN_MATCHES:
                                    plate_counts: Dict[str, int] = {}
                                    for attempt in track_info['lp_ocr_attempts']:
                                        if attempt.get('text'): 
                                            plate_counts[attempt['text']] = plate_counts.get(attempt['text'], 0) + 1
                                    
                                    for plate_text_candidate, count in plate_counts.items():
                                        if count >= settings.OCR_CONFIRMATION_MIN_MATCHES:
                                            track_info['lp_text'] = plate_text_candidate
                                            for attempt_detail in reversed(track_info['lp_ocr_attempts']):
                                                if attempt_detail.get('text') == plate_text_candidate:
                                                    track_info['lp_confidence'] = attempt_detail.get('confidence')
                                                    track_info['lp_bbox_local'] = attempt_detail.get('bbox')
                                                    break
                                            track_info['lp_confirmed'] = True
                                            track_info['lp_display_text'] = plate_text_candidate
                                            track_info['lp_ocr_attempts'].clear()
                                            confirmed_lp_texts_set.add(plate_text_candidate)
                                            print(f"[CONFIRMED] Track ID {track_id_from_worker}: LP '{plate_text_candidate}' confirmed.")

                                            try:
                                                vehicle_image_base64_for_api: Optional[str] = None
                                                vehicle_bbox_coords = track_info.get('vehicle_bbox_coords') 
                                                if vehicle_bbox_coords and original_frame is not None:
                                                    # vehicle_bbox_coords is x1,y1,x2,y2 from main loop detections
                                                    x1_veh, y1_veh, x2_veh, y2_veh = vehicle_bbox_coords
                                                    
                                                    y_end = min(y2_veh, original_frame.shape[0])
                                                    x_end = min(x2_veh, original_frame.shape[1])
                                                    y_start = max(0, y1_veh)
                                                    x_start = max(0, x1_veh)

                                                    if y_end > y_start and x_end > x_start:
                                                        vehicle_crop_img = original_frame[y_start:y_end, x_start:x_end]
                                                        if vehicle_crop_img.size > 0:
                                                            try:
                                                                is_success_veh, buffer_veh = cv2.imencode(".jpg", vehicle_crop_img)
                                                                if is_success_veh:
                                                                    vehicle_image_base64_for_api = base64.b64encode(buffer_veh).decode('utf-8')
                                                            except Exception as e_v_encode:
                                                                print(f"[API_CLIENT_ERROR] Failed to encode vehicle image for Track ID {track_id_from_worker}: {e_v_encode}")
                                                
                                                lp_data_for_api = LicensePlateData(
                                                    text=track_info.get('lp_text'),
                                                    lp_detection_confidence=track_info.get('lp_confidence'),
                                                    detected=True, 
                                                    image_base64=track_info.get('lp_image_base64') 
                                                )
                                                attributes_for_api = VehicleAttributes(
                                                    license_plate=lp_data_for_api,
                                                    vehicle_image_base64=vehicle_image_base64_for_api 
                                                )
                                                
                                                # Convert vehicle_bbox_coords from [x1,y1,x2,y2] to [x,y,w,h] for API
                                                api_bbox_coords = [0,0,0,0]
                                                if track_info.get('vehicle_bbox_coords'):
                                                    x1,y1,x2,y2 = track_info['vehicle_bbox_coords']
                                                    api_bbox_coords = [x1, y1, x2-x1, y2-y1]

                                                vehicle_event_data = VehicleEvent(
                                                    timestamp_utc=datetime.datetime.utcnow(), 
                                                    camera_id=settings.CAMERA_ID,
                                                    vehicle_track_id=track_id_from_worker,
                                                    base_class=track_info.get('vehicle_class', 'unknown'), 
                                                    confidence_base_class=track_info.get('vehicle_confidence', 0.0), 
                                                    bounding_box_frame_coords=api_bbox_coords, 
                                                    attributes=attributes_for_api
                                                )
                                                
                                                response = requests.post(settings.API_ENDPOINT_URL, json=vehicle_event_data.model_dump(mode='json'))
                                                if response.status_code == 200:
                                                    print(f"[API_CLIENT] Successfully sent event for Track ID {track_id_from_worker} to API.")
                                                    # Store event in Qdrant
                                                    try:
                                                        embed_and_store_event(event=vehicle_event_data, sqlite_event_id=None)
                                                        print(f"[QDRANT_CLIENT] Successfully stored event for Track ID {track_id_from_worker} in Qdrant.")
                                                    except Exception as e_qdrant:
                                                        print(f"[QDRANT_CLIENT_ERROR] Failed to store event for Track ID {track_id_from_worker} in Qdrant: {e_qdrant}")
                                                else:
                                                    print(f"[API_CLIENT_ERROR] Failed to send event for Track ID {track_id_from_worker}. Status: {response.status_code}, Response: {response.text}")
                                            except requests.exceptions.RequestException as e_req:
                                                print(f"[API_CLIENT_ERROR] Request failed for Track ID {track_id_from_worker}: {e_req}")
                                            except Exception as e_main_event_block: # Catch other potential errors in this block
                                                print(f"[PIPELINE_ERROR] Error during event finalization/posting/Qdrant for Track ID {track_id_from_worker}: {e_main_event_block}")
                                            break
                            
                            if not track_info['lp_confirmed']:
                                current_num_attempts = len(track_info['lp_ocr_attempts'])
                                last_successful_attempt_text = None
                                last_successful_attempt_conf = None
                                last_successful_attempt_bbox = None

                                for past_attempt in reversed(track_info['lp_ocr_attempts']):
                                    if past_attempt.get("text"):
                                        last_successful_attempt_text = past_attempt["text"]
                                        last_successful_attempt_conf = past_attempt.get("confidence")
                                        last_successful_attempt_bbox = past_attempt.get("bbox")
                                        break
                                
                                if last_successful_attempt_text:
                                    track_info['lp_display_text'] = f"{last_successful_attempt_text} ({current_num_attempts}/{settings.OCR_CONFIRMATION_ATTEMPTS})"
                                    if last_successful_attempt_conf is not None: track_info['lp_confidence'] = last_successful_attempt_conf
                                    if last_successful_attempt_bbox is not None: track_info['lp_bbox_local'] = last_successful_attempt_bbox
                                elif lp_detected_flag:
                                    status_msg = "Scan" if not text_attempt else "OCR Fail"
                                    track_info['lp_display_text'] = f"LP ({status_msg} {current_num_attempts}/{settings.OCR_CONFIRMATION_ATTEMPTS})"
                                    if conf_attempt is not None: track_info['lp_confidence'] = conf_attempt
                                    if bbox_attempt is not None: track_info['lp_bbox_local'] = bbox_attempt
                                else:
                                    track_info['lp_display_text'] = ""
                                    track_info['lp_confidence'] = 0.0
            except queue.Empty: pass

            if vehicle_results and vehicle_results[0].boxes is not None:
                total_vehicles_detected_main += len(vehicle_results[0].boxes)
                for box in vehicle_results[0].boxes: 
                    cls_id, conf_v = int(box.cls.item()), float(box.conf.item())
                    class_name = vehicle_names.get(cls_id, "Unknown")
                    if not active_filters.get(class_name, False): continue

                    x1_inf,y1_inf,x2_inf,y2_inf = map(int, box.xyxy[0])
                    track_id = int(box.id.item()) if box.id is not None else None
                    
                    x1_v,y1_v,x2_v,y2_v = int(x1_inf/s_x),int(y1_inf/s_y),int(x2_inf/s_x),int(y2_inf/s_y)

                    current_frame_detections_for_api.append({
                        "track_id": track_id,
                        "class_name": class_name,
                        "confidence": round(conf_v, 4),
                        "box_coords": [x1_v, y1_v, x2_v, y2_v] 
                    })
                    
                    if track_id:
                        unique_vehicle_track_ids.add(track_id)
                        if track_id not in tracked_vehicle_data:
                            tracked_vehicle_data[track_id] = {
                                'lp_text': None, 'lp_confidence': 0.0, 'lp_bbox_local': None,
                                'lp_confirmed': False, 'lp_ocr_attempts': [],
                                'lp_last_ocr_frame': 0, 'lp_display_text': "", 'lp_image_base64': None,
                                'vehicle_class': class_name, 
                                'vehicle_confidence': round(conf_v,4),
                                'vehicle_bbox_coords': [x1_v, y1_v, x2_v, y2_v] 
                            }
                        else: 
                            tracked_vehicle_data[track_id]['vehicle_class'] = class_name
                            tracked_vehicle_data[track_id]['vehicle_confidence'] = round(conf_v,4)
                            tracked_vehicle_data[track_id]['vehicle_bbox_coords'] = [x1_v, y1_v, x2_v, y2_v]


                    label_v = f"ID:{track_id} {class_name} {conf_v:.2f}" if track_id else f"{class_name} {conf_v:.2f}"
                    if gui_works:
                        draw_text_with_background(display_image, label_v, (x1_v, y1_v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, settings.VEHICLE_BOX_COLOR, settings.TEXT_BG_COLOR)
                        cv2.rectangle(display_image,(x1_v,y1_v),(x2_v,y2_v),settings.VEHICLE_BOX_COLOR,2)
                    
                    db_record_base_dict = {
                        'timestamp': datetime.datetime.now().isoformat(), 'frame_number': frames_processed,
                        'track_id': track_id, 'vehicle_class': class_name,
                        'vehicle_confidence': round(conf_v,4), 'vehicle_bbox': f"{x1_v},{y1_v},{x2_v},{y2_v}",
                        'lp_confidence': None, 'lp_bbox_local': None, 'lp_text': None
                    }

                    perform_lp_ocr = False
                    if track_id and lp_model_instance:
                        current_track_info = tracked_vehicle_data[track_id]
                        cooldown_period = settings.CONFIRMED_LP_COOLDOWN_FRAMES if current_track_info['lp_confirmed'] else settings.LP_OCR_COOLDOWN_FRAMES
                        if frames_processed >= current_track_info['lp_last_ocr_frame'] + cooldown_period:
                            perform_lp_ocr = True
                    
                    if perform_lp_ocr:
                        vehicle_crop_for_q = original_frame[y1_v:y2_v, x1_v:x2_v] 
                        if vehicle_crop_for_q.size > 0:
                            try:
                                ocr_task_queue.put_nowait((vehicle_crop_for_q.copy(), db_record_base_dict, track_id, frames_processed))
                                total_lp_ocr_tasks_queued += 1
                                tracked_vehicle_data[track_id]['lp_last_ocr_frame'] = frames_processed
                                _post_backend_status("pipeline_main_loop", "ocr_task_queued", {"track_id": track_id, "frame_num": frames_processed, "queue_size": ocr_task_queue.qsize()})
                            except queue.Full:
                                ocr_queue_full_count += 1
                                _post_backend_status("pipeline_main_loop", "ocr_queue_full", {"track_id": track_id, "frame_num": frames_processed})
                    
                    if gui_works and track_id and track_id in tracked_vehicle_data:
                        track_info_display = tracked_vehicle_data[track_id]
                        lp_text_to_show = track_info_display.get('lp_display_text', "")
                        
                        if lp_text_to_show:
                            current_lp_color = settings.LP_TEXT_CONFIRMED_COLOR if track_info_display['lp_confirmed'] else settings.LP_TEXT_COLOR
                            lp_display_y = y1_v + 30
                            text_org_lp = (x1_v, lp_display_y)
                            
                            display_string = ""
                            if track_info_display['lp_confirmed']:
                                display_string = f"LP: {track_info_display['lp_text']} (C)"
                            elif lp_text_to_show:
                                current_lp_det_conf = track_info_display.get('lp_confidence', 0.0) 
                                if isinstance(current_lp_det_conf, float) and current_lp_det_conf > 0 and "Scan" not in lp_text_to_show and "Fail" not in lp_text_to_show:
                                    display_string = f"LP({current_lp_det_conf*100:.0f}%): {lp_text_to_show}"
                                else:
                                    display_string = f"LP: {lp_text_to_show}"

                            if display_ocr_on_gui and display_string:
                                 draw_text_with_background(display_image, display_string, text_org_lp, cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_lp_color, settings.TEXT_BG_COLOR)

                            lp_bbox_to_draw = track_info_display.get('lp_bbox_local')
                            if lp_bbox_to_draw:
                                lp_rel_x, lp_rel_y, lp_w, lp_h = lp_bbox_to_draw
                                gx1_abs = x1_v + lp_rel_x
                                gy1_abs = y1_v + lp_rel_y
                                gx2_abs = gx1_abs + lp_w
                                gy2_abs = gy1_abs + lp_h

                                gx1_clipped = max(x1_v, gx1_abs)
                                gy1_clipped = max(y1_v, gy1_abs)
                                gx2_clipped = min(x2_v, gx2_abs)
                                gy2_clipped = min(y2_v, gy2_abs)

                                if gx2_clipped > gx1_clipped and gy2_clipped > gy1_clipped:
                                    cv2.rectangle(display_image, (gx1_clipped, gy1_clipped), (gx2_clipped, gy2_clipped), settings.LP_BOX_COLOR, 2)
                        elif lp_model_instance is None and track_id:
                             lp_display_y = y1_v + 30
                             text_org_lp = (x1_v, lp_display_y)
                             draw_text_with_background(display_image, "LP N/A", text_org_lp, cv2.FONT_HERSHEY_SIMPLEX, 0.4, settings.LP_MODEL_NA_COLOR, settings.TEXT_BG_COLOR)
            
            loop_processing_time = time.time() - loop_start_time
            total_loop_time += loop_processing_time
            current_fps = 1.0 / loop_processing_time if loop_processing_time > 0 else 0

            fps_text = f"FPS: {current_fps:.2f}"
            draw_text_with_background(live_feed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.FPS_TEXT_COLOR, settings.TEXT_BG_COLOR, thickness=2)
            if gui_works: 
                 draw_text_with_background(display_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, settings.FPS_TEXT_COLOR, settings.TEXT_BG_COLOR, thickness=2)

            if live_frame_api_available and frames_processed % settings.LIVE_FRAME_SEND_INTERVAL_MAIN_LOOPS == 0:
                if live_feed_frame is not None and live_feed_frame.size > 0:
                    try:
                        is_success_frame, buffer_frame = cv2.imencode(".jpg", live_feed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if is_success_frame:
                            frame_b64_str = base64.b64encode(buffer_frame).decode('utf-8')
                            
                            frame_payload = {
                                "frame_base64": frame_b64_str,
                                "detections": current_frame_detections_for_api
                            }
                            try:
                                requests.post(settings.API_LIVE_FRAME_UPDATE_URL, json=frame_payload, timeout=0.2)
                                live_frame_connection_failures = 0 # Reset on success
                            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e_connect:
                                live_frame_connection_failures += 1
                                print(f"[PIPELINE_WARN] Failed to send live frame to API (Attempt {live_frame_connection_failures}/{MAX_LIVE_FRAME_FAILURES}): {e_connect}")
                                if live_frame_connection_failures >= MAX_LIVE_FRAME_FAILURES:
                                    live_frame_api_available = False
                                    print("[PIPELINE_ERROR] Max retries exceeded. Disabling live frame sending to API. Ensure API server is running.")
                            except requests.exceptions.RequestException as e_frame_req:
                                # For other request exceptions, don't necessarily disable, but log
                                print(f"[PIPELINE_WARN] Failed to send live frame to API due to other request error: {e_frame_req}")
                        else:
                            print("[PIPELINE_WARN] Failed to encode display_image for live feed API.")
                    except Exception as e_frame_encode:
                        print(f"[PIPELINE_ERROR] Error encoding/sending live frame: {e_frame_encode}")
            
            if gui_works:
                try:
                    max_disp_w, max_disp_h = 2560, 1440
                    h_img, w_img = display_image.shape[:2]
                    s = min(max_disp_w/w_img, max_disp_h/h_img) if w_img>0 and h_img>0 else 1
                    disp_img_resized = cv2.resize(display_image,(int(w_img*s), int(h_img*s))) if s < 1 else display_image
                    cv2.imshow(window_name, disp_img_resized)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): print("\n'q' pressed, exiting."); break
                    elif key == ord('o'):
                        display_ocr_on_gui = not display_ocr_on_gui
                        print(f"[INFO] OCR Display Toggled: {'ON' if display_ocr_on_gui else 'OFF'}")
                    elif key == ord('p'):
                        ENABLE_LP_PREPROCESSING = not ENABLE_LP_PREPROCESSING 
                        print(f"[INFO] LP Preprocessing Toggled: {'ON' if ENABLE_LP_PREPROCESSING else 'OFF'}")
                    elif key in filter_keys_map:
                        cls_toggle = filter_keys_map[key]
                        active_filters[cls_toggle] = not active_filters[cls_toggle]
                        print(f"[INFO] Filter for {cls_toggle} set to: {active_filters[cls_toggle]}")
                except cv2.error as e_cv:
                    if frames_processed == 1: print(f"[WARN] OpenCV GUI error: {e_cv}. Switching to headless.")
                    gui_works = False
            
            if frames_processed % 100 == 0:
                 print(f"[INFO] Frame {frames_processed}, FPS: {current_fps:.2f}{' (Headless)' if not gui_works else ''}")
    
    except KeyboardInterrupt: print("\n[INFO] Pipeline interrupted by user.")
    except Exception as e_pipe: print(f"[ERROR] Pipeline exception in run_main_pipeline: {e_pipe}")
    finally:
        print("[INFO] Main loop finished. Signaling OCR worker to stop...")
        ocr_task_queue.put(None) 
        ocr_thread.join(timeout=5) 
        if ocr_thread.is_alive(): print("[WARN] OCR worker thread did not stop in time.")

        overall_duration = time.time() - start_time_overall
        avg_fps = frames_processed / total_loop_time if total_loop_time > 0 else 0
        print(f"\n--- Final Summary & Metrics ---")
        print(f"Total frames processed by main loop: {frames_processed}")
        print(f"Total script duration: {overall_duration:.2f}s")
        print(f"Average FPS (main loop): {avg_fps:.2f}")
        print(f"Total unique vehicle track IDs encountered: {len(unique_vehicle_track_ids)}")
        print(f"Total vehicle detections by main model: {total_vehicles_detected_main}")
        print(f"Total LP/OCR tasks queued for worker: {total_lp_ocr_tasks_queued}")
        print(f"OCR task queue full (tasks skipped): {ocr_queue_full_count} times")
        print(f"Worker - Tasks processed: {worker_stats_ref.get('tasks_processed', 0)}")
        print(f"Worker - Successful LP detections: {worker_stats_ref.get('lp_detections', 0)}")
        print(f"Worker - Successful OCR reads: {worker_stats_ref.get('ocr_successes', 0)}")
        print(f"Total unique LPs confirmed: {len(confirmed_lp_texts_set)}") 
        print("-----------------------------------\n")

        if cap.isOpened(): cap.release()
        if gui_works:
            try: cv2.destroyAllWindows()
            except cv2.error: pass
        elif frames_processed > 0 : print("[INFO] Ran in headless mode due to OpenCV GUI issue.")

# Note: The if __name__ == "__main__": block will be moved to main.py later.
# For now, nurvek-v.py will be modified to import and call run_main_pipeline from pipeline.py
# after this step is complete and nurvek-v.py's current pipeline functions are removed.