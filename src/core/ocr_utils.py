import cv2
import numpy as np
from typing import Tuple, Any, Optional, List
import easyocr # Added for EasyOCR
# import requests # No longer needed for Ollama
# import base64 # No longer needed for Ollama
# import io # For potential future use with image bytes, though direct base64 is fine
# import json # No longer needed for Ollama

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

# Ollama-based functions removed
# def _extract_text_from_vision_model_response ...
# async def extract_plate_text_with_vision_model_async ...
# def extract_plate_text_with_vision_model_sync ...

def extract_license_plate_info_ocr(
    vehicle_crop_cv2_img: ImageType,
    lp_detection_model: Any,  # YOLO model instance
    ocr_reader_instance: easyocr.Reader # EasyOCR reader instance
) -> Tuple[bool, bool, Optional[str], float, Optional[BBoxType], Optional[str], Optional[ImageType]]:
    """
    Detects license plate, performs OCR using EasyOCR, and returns results.
    Returns:
        lp_detected (bool): If a license plate was detected.
        ocr_success (bool): If OCR successfully read a valid plate.
        ocr_text (Optional[str]): The validated OCR text.
        lp_confidence (float): Confidence score of the LP detection.
        lp_bbox_local_to_vehicle_crop (Optional[BBoxType]): BBox of LP relative to vehicle crop.
        raw_ocr_text (Optional[str]): Raw text from OCR before validation.
        lp_image_sent_to_ocr (Optional[ImageType]): The actual image crop sent to OCR.
    """
    ocr_text_result: Optional[str] = None
    lp_detected_flag: bool = False
    ocr_success_flag: bool = False
    lp_detection_confidence: float = 0.0
    lp_bbox_coords: Optional[BBoxType] = None # Relative to vehicle_crop_cv2_img
    raw_easyocr_text: Optional[str] = None
    image_sent_to_ocr: Optional[ImageType] = None

    if vehicle_crop_cv2_img is None or vehicle_crop_cv2_img.size == 0:
        return lp_detected_flag, ocr_success_flag, ocr_text_result, lp_detection_confidence, lp_bbox_coords, raw_easyocr_text, image_sent_to_ocr

    if lp_detection_model is None or ocr_reader_instance is None:
        # print("[OCR_UTIL_WARN] LP detection model or OCR reader not provided.")
        return lp_detected_flag, ocr_success_flag, ocr_text_result, lp_detection_confidence, lp_bbox_coords, raw_easyocr_text, image_sent_to_ocr

    # 1. Detect License Plate
    lp_results = lp_detection_model.predict(vehicle_crop_cv2_img, verbose=False, conf=settings.LP_CONFIDENCE_THRESHOLD, half=True)

    best_lp_crop: Optional[ImageType] = None

    if lp_results and lp_results[0].boxes and len(lp_results[0].boxes) > 0:
        # Find the LP with the highest confidence
        best_box_for_ocr = None
        max_conf = 0.0
        for box in lp_results[0].boxes:
            cls_id, conf = int(box.cls.item()), float(box.conf.item())
            if cls_id == 0: # Assuming class 0 is 'license_plate'
                if conf > max_conf:
                    max_conf = conf
                    best_box_for_ocr = box

        if best_box_for_ocr is not None:
            lp_detected_flag = True
            lp_detection_confidence = round(max_conf, 4)
            x1, y1, x2, y2 = map(int, best_box_for_ocr.xyxy[0])
            lp_bbox_coords = [x1, y1, x2 - x1, y2 - y1] # Store as x,y,w,h

            # Ensure coordinates are within bounds of the vehicle crop
            y1_c, y2_c = max(0, y1), min(vehicle_crop_cv2_img.shape[0], y2)
            x1_c, x2_c = max(0, x1), min(vehicle_crop_cv2_img.shape[1], x2)

            if y2_c > y1_c and x2_c > x1_c:
                best_lp_crop = vehicle_crop_cv2_img[y1_c:y2_c, x1_c:x2_c]

    # 2. Perform OCR if LP detected
    if lp_detected_flag and best_lp_crop is not None:
        image_to_ocr = best_lp_crop
        if settings.ENABLE_LP_PREPROCESSING:
            preprocessed_lp = preprocess_lp_for_ocr(best_lp_crop)
            if preprocessed_lp is not None:
                image_to_ocr = preprocessed_lp
        
        image_sent_to_ocr = image_to_ocr.copy() # Store the image actually sent

        try:
            # EasyOCR's readtext with detail=0 returns a list of strings
            ocr_results_list = ocr_reader_instance.readtext(
                image_to_ocr,
                allowlist=settings.EASYOCR_ALLOWLIST,
                detail=0, # Returns list of strings
                paragraph=False # Treat each line as a separate text block
            )
            
            if ocr_results_list:
                # Concatenate results and clean, or pick the best one.
                # For simplicity, let's concatenate and then validate.
                # A more sophisticated approach might analyze confidence if detail=1 was used.
                concatenated_text = "".join(ocr_results_list).upper()
                raw_easyocr_text = concatenated_text # Store the raw concatenated (but cleaned char-wise by allowlist)
                
                # Validate length
                if len(raw_easyocr_text) in settings.VALID_LP_LENGTHS:
                    ocr_text_result = raw_easyocr_text
                    ocr_success_flag = True
                else:
                    # print(f"[OCR_UTIL_INFO] EasyOCR text '{raw_easyocr_text}' not a valid LP length.")
                    pass # Not a fatal error, just not a "successful" validated read
            else:
                # print("[OCR_UTIL_INFO] EasyOCR returned no text.")
                pass

        except Exception as e:
            print(f"[OCR_UTIL_ERROR] EasyOCR failed: {e}")
            # ocr_success_flag remains False
    
    elif lp_detected_flag and best_lp_crop is None:
        # This case should ideally not happen if lp_detected_flag is true
        # print("[OCR_UTIL_WARN] LP detected but crop was None.")
        pass

    return lp_detected_flag, ocr_success_flag, ocr_text_result, lp_detection_confidence, lp_bbox_coords, raw_easyocr_text, image_sent_to_ocr