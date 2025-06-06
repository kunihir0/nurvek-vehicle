import cv2
import numpy as np
from typing import Tuple
from src.config import settings # For TEXT_BG_ALPHA

# Define type hints for common OpenCV structures if not already globally available
# For simplicity, we'll use basic types here or assume cv2 types are understood.
# More specific types like np.ndarray could be used.
ImageType = np.ndarray 
PointType = Tuple[int, int]
ColorType = Tuple[int, int, int]

def draw_text_with_background(
    img: ImageType, 
    text: str, 
    org: PointType, 
    font_face: int, 
    font_scale: float, 
    text_color: ColorType, 
    bg_color: ColorType, 
    thickness: int = 1, 
    line_type: int = cv2.LINE_AA, 
    pad: int = 3
) -> None:
    """Draws text with a semi-transparent background on an image."""
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    
    # Top-left corner of the background rectangle
    bg_tl_x = org[0] - pad
    bg_tl_y = org[1] + baseline - text_height - pad
    # Bottom-right corner of the background rectangle
    bg_br_x = org[0] + text_width + pad
    bg_br_y = org[1] + baseline + pad

    img_h, img_w = img.shape[:2]
    
    # Clip coordinates to be within image boundaries
    rect_tl_x_clipped = max(0, bg_tl_x)
    rect_tl_y_clipped = max(0, bg_tl_y)
    rect_br_x_clipped = min(img_w, bg_br_x)
    rect_br_y_clipped = min(img_h, bg_br_y)

    if rect_br_x_clipped > rect_tl_x_clipped and rect_br_y_clipped > rect_tl_y_clipped:
        sub_img = img[rect_tl_y_clipped:rect_br_y_clipped, rect_tl_x_clipped:rect_br_x_clipped]
        
        bg_rect = np.full(sub_img.shape, bg_color, dtype=np.uint8)
        
        # Blend the background rectangle with the sub-image
        # Access TEXT_BG_ALPHA from the imported settings
        res = cv2.addWeighted(sub_img, 1 - settings.TEXT_BG_ALPHA, bg_rect, settings.TEXT_BG_ALPHA, 0)
        
        img[rect_tl_y_clipped:rect_br_y_clipped, rect_tl_x_clipped:rect_br_x_clipped] = res

    cv2.putText(img, text, org, font_face, font_scale, text_color, thickness, line_type)