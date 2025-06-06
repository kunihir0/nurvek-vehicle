import cv2
import numpy as np
from typing import Tuple, Any, Optional, List
import requests
import base64
import io # For potential future use with image bytes, though direct base64 is fine
import json # For parsing streamed JSON lines

from src.config import settings # Import settings

# Define type hints for clarity
ImageType = np.ndarray
BBoxType = List[int] # Typically [x, y, w, h] or [x1, y1, x2, y2]

def preprocess_lp_for_ocr(image_crop: ImageType) -> Optional[ImageType]:
    """Converts LP crop to grayscale and applies adaptive thresholding."""
    if image_crop is None or image_crop.size == 0:
        return None
    
    gray_lp = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    
    block_size = 11 
    C_val = 5
    binary_lp = cv2.adaptiveThreshold(gray_lp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                      cv2.THRESH_BINARY_INV, block_size, C_val)
    return binary_lp

def _extract_text_from_vision_model_response(response_json: dict) -> Optional[str]:
    """Helper to extract text from Ollama's /api/chat response structure."""
    try:
        # For /api/chat (non-streaming)
        if response_json.get("message") and response_json["message"].get("content"):
            return response_json["message"]["content"].strip()
        # For /api/generate (non-streaming, if it were used and returned full JSON)
        # elif response_json.get("response"):
        #     return response_json.get("response").strip()
    except Exception as e:
        print(f"[VISION_OCR_UTIL_ERROR] Error parsing vision model response: {e}, Response: {response_json}")
    return None


async def extract_plate_text_with_vision_model_async(lp_image_crop_np: ImageType) -> Optional[str]:
    """
    Performs OCR on a license plate crop using the configured vision model via Ollama.
    (Async version - not currently used by synchronous pipeline but good for future)
    """
    if lp_image_crop_np is None or lp_image_crop_np.size == 0:
        return None

    try:
        is_success, buffer = cv2.imencode(".jpg", lp_image_crop_np)
        if not is_success:
            print("[VISION_OCR_ERROR] Failed to encode LP image to JPG.")
            return None
        
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "model": settings.VISION_OCR_MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": settings.VISION_OCR_PROMPT,
                    "images": [img_base64]
                }
            ],
            "stream": False # Non-streaming for direct response
        }
        
        # This would need an async HTTP client like aiohttp if called from async code
        response = requests.post(
            settings.OLLAMA_API_BASE_URL, 
            json=payload, 
            timeout=settings.VISION_OCR_TIMEOUT_SECONDS
        )
        response.raise_for_status() # Raise an exception for HTTP errors
        
        response_data = response.json()
        extracted_text = _extract_text_from_vision_model_response(response_data)

        if extracted_text and extracted_text.upper() != "UNKNOWN_PLATE":
            # Basic cleaning, similar to EasyOCR allowlist effect
            cleaned_text = "".join(filter(str.isalnum, extracted_text)).upper()
            return cleaned_text
        elif extracted_text and extracted_text.upper() == "UNKNOWN_PLATE":
            return None # Model explicitly said unknown
        else:
            print(f"[VISION_OCR_WARN] Vision model returned no usable text or indicated UNKNOWN_PLATE. Raw: {extracted_text}")
            return None

    except requests.exceptions.Timeout:
        print(f"[VISION_OCR_ERROR] Timeout connecting to Ollama API at {settings.OLLAMA_API_BASE_URL}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[VISION_OCR_ERROR] Request to Ollama API failed: {e}")
        return None
    except Exception as e:
        print(f"[VISION_OCR_ERROR] General error in extract_plate_text_with_vision_model: {e}")
        return None

def extract_plate_text_with_vision_model_sync(lp_image_crop_np: ImageType) -> Optional[str]:
    """
    Synchronous version of OCR using the vision model.
    """
    if lp_image_crop_np is None or lp_image_crop_np.size == 0:
        return None

    try:
        is_success, buffer = cv2.imencode(".jpg", lp_image_crop_np)
        if not is_success:
            print("[VISION_OCR_ERROR_SYNC] Failed to encode LP image to JPG.")
            return None
        
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        payload = {
            "model": settings.VISION_OCR_MODEL_NAME,
            "messages": [ # Using /api/chat structure
                {
                    "role": "user",
                    "content": settings.VISION_OCR_PROMPT,
                    "images": [img_base64]
                }
            ],
            "stream": True # Changed to True for streaming
        }
        
        full_extracted_text = ""
        accumulated_raw_stream_for_api = ""

        response = requests.post(
            settings.OLLAMA_API_BASE_URL,
            json=payload,
            timeout=settings.VISION_OCR_TIMEOUT_SECONDS,
            stream=True # Ensure requests library handles it as a stream
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                try:
                    decoded_line = line.decode('utf-8')
                    # Send chunk to internal API before further processing
                    # This sends each raw JSON line from Ollama's stream
                    try:
                        requests.post(settings.API_INTERNAL_OCR_STREAM_UPDATE_URL,
                                      json={"ocr_chunk": decoded_line, "source": "vision_ocr_stream"},
                                      timeout=0.5) # Short timeout, fire and forget
                    except Exception as e_api_post:
                        print(f"[VISION_OCR_STREAM_POST_ERROR] Failed to post chunk to internal API: {e_api_post}")

                    # Process the chunk for assembling the full response for the pipeline
                    json_chunk = json.loads(decoded_line)
                    if json_chunk.get("message") and json_chunk["message"].get("content"):
                        full_extracted_text += json_chunk["message"]["content"]
                    # Check for Ollama's own 'done' flag within the stream object if it's the /api/chat format
                    if json_chunk.get("done") and json_chunk.get("done_reason") == "stop": # For /api/chat like structure
                        break
                    # For /api/generate, the 'done' flag is usually on the last object.
                    # The loop will naturally end when iter_lines is exhausted for /api/generate.
                    # If using /api/generate, the final object might have a "response" field with the full text.
                    # The current Ollama endpoint in settings is /api/chat, so this structure is fine.

                except json.JSONDecodeError:
                    print(f"[VISION_OCR_WARN_SYNC] Non-JSON line in stream: {line}")
                except Exception as e_stream_proc:
                    print(f"[VISION_OCR_ERROR_SYNC] Error processing stream line: {e_stream_proc}")
        
        # After loop, one last post of the *assembled* text if it's meaningful
        # This is more for a summary or if the API expects a final "full" version.
        # The per-chunk posting above handles the live stream.
        # For now, we won't send an additional "final_chunk" of the assembled text,
        # as the frontend will assemble it from the chunks.

        if full_extracted_text and full_extracted_text.upper() != "UNKNOWN_PLATE":
            cleaned_text = "".join(filter(str.isalnum, full_extracted_text)).upper()
            return cleaned_text
        elif full_extracted_text and full_extracted_text.upper() == "UNKNOWN_PLATE":
            return None
        else:
            print(f"[VISION_OCR_WARN_SYNC] Vision model returned no usable text or indicated UNKNOWN_PLATE. Raw: {full_extracted_text}")
            return None

    except requests.exceptions.Timeout:
        print(f"[VISION_OCR_ERROR_SYNC] Timeout connecting to Ollama API at {settings.OLLAMA_API_BASE_URL}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[VISION_OCR_ERROR_SYNC] Request to Ollama API failed: {e}")
        return None
    except Exception as e:
        print(f"[VISION_OCR_ERROR_SYNC] General error in extract_plate_text_with_vision_model_sync: {e}")
        return None

def extract_license_plate_info_ocr( # Renamed from extract_license_plate_info_ocr
    vehicle_crop_cv2_img: ImageType, 
    lp_detection_model: Any, # YOLO model instance
    # ocr_reader_instance: Any, # No longer needed
    enable_preprocessing: bool 
) -> Tuple[bool, bool, Optional[str], float, Optional[BBoxType], Optional[str], Optional[ImageType]]:
    """
    Detects license plate, performs OCR using Vision Model, and returns results.
    """
    ocr_text_result: Optional[str] = None
    lp_detected_flag: bool = False
    ocr_success_flag: bool = False
    best_plate_score: float = 0.0
    best_plate_box_coords: Optional[BBoxType] = None
    raw_cleaned_ocr_text: Optional[str] = None # This will be the direct output from vision model
    final_lp_image_for_ocr: Optional[ImageType] = None

    if vehicle_crop_cv2_img.size == 0:
        return lp_detected_flag, ocr_success_flag, ocr_text_result, best_plate_score, best_plate_box_coords, raw_cleaned_ocr_text, final_lp_image_for_ocr
    
    if lp_detection_model is None:
        return lp_detected_flag, ocr_success_flag, ocr_text_result, best_plate_score, best_plate_box_coords, raw_cleaned_ocr_text, final_lp_image_for_ocr

    raw_plate_crop: Optional[ImageType] = None
    lp_results = lp_detection_model.predict(vehicle_crop_cv2_img, verbose=False, conf=settings.LP_CONFIDENCE_THRESHOLD, half=True)

    if lp_results and lp_results[0].boxes and len(lp_results[0].boxes) > 0:
        for box in lp_results[0].boxes:
            raw_cls_id, raw_conf = int(box.cls.item()), float(box.conf.item())
            if raw_cls_id == 0 and raw_conf > best_plate_score: 
                best_plate_score = raw_conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                best_plate_box_coords = [x1, y1, x2 - x1, y2 - y1] 
                
                y1_o, y2_o = max(0, y1), min(vehicle_crop_cv2_img.shape[0], y2)
                x1_o, x2_o = max(0, x1), min(vehicle_crop_cv2_img.shape[1], x2)
                if y2_o > y1_o and x2_o > x1_o:
                    raw_plate_crop = vehicle_crop_cv2_img[y1_o:y2_o, x1_o:x2_o]
        
        if best_plate_box_coords:
            lp_detected_flag = True
            if raw_plate_crop is not None:
                final_lp_image_for_ocr = raw_plate_crop 
                if enable_preprocessing: 
                    preprocessed_result = preprocess_lp_for_ocr(raw_plate_crop)
                    if preprocessed_result is not None:
                        final_lp_image_for_ocr = preprocessed_result
                
                # Perform OCR using the vision model
                extracted_text = extract_plate_text_with_vision_model_sync(final_lp_image_for_ocr)
                raw_cleaned_ocr_text = extracted_text # Store the direct result

                if extracted_text and len(extracted_text) in settings.VALID_LP_LENGTHS:
                    ocr_text_result = extracted_text
                    ocr_success_flag = True
                elif extracted_text: # Text extracted but not valid length
                    print(f"[VISION_OCR_INFO] Extracted text '{extracted_text}' not a valid LP length.")
            
            # Ensure final_lp_image_for_ocr is set if LP was detected, even if OCR fails
            elif raw_plate_crop is not None and final_lp_image_for_ocr is None:
                 final_lp_image_for_ocr = raw_plate_crop
                    
    return lp_detected_flag, ocr_success_flag, ocr_text_result, best_plate_score, best_plate_box_coords, raw_cleaned_ocr_text, final_lp_image_for_ocr