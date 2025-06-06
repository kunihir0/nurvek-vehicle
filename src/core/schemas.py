from typing import Optional, List, Any
from pydantic import BaseModel, Field
import datetime

class LicensePlateData(BaseModel):
    text: Optional[str] = None
    # Using alias to match the JSON example from plan.md, 
    # but internally it's clear this is LP detection confidence.
    lp_detection_confidence: Optional[float] = Field(None, alias="ocrConfidence") 
    detected: bool = False
    image_base64: Optional[str] = None # Base64 encoded image of the LP crop
    # We can add ocr_confidence if easyocr provides it or if we derive it later
    # ocr_processing_confidence: Optional[float] = None

class VehicleAttributes(BaseModel):
    license_plate: LicensePlateData = Field(default_factory=LicensePlateData)
    vehicle_image_base64: Optional[str] = None # Base64 encoded image of the vehicle crop
    # color: Optional[ColorData] = None # Future
    # damage_state: Optional[DamageData] = None # Future
    # model_type: Optional[ModelData] = None # Future

class VehicleEvent(BaseModel):
    timestamp_utc: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    camera_id: str
    vehicle_track_id: int
    base_class: str
    confidence_base_class: float
    bounding_box_frame_coords: List[int] # x, y, w, h (as per current DB logging)
    # world_coordinates_estimate_gta: Optional[List[float]] = None # Future
    attributes: VehicleAttributes = Field(default_factory=VehicleAttributes)
    # current_speed_estimate_kph: Optional[float] = None # Future
    # heading_degrees: Optional[float] = None # Future
    # flags: Optional[List[str]] = None # Future

    class Config:
        # Ensure that when we convert to JSON, datetime objects are handled correctly
        # Pydantic v2 uses model_dump_json which handles this by default.
        # For older Pydantic, json_encoders might be needed.
        # For now, this is fine.
        pass

# Example usage (for testing, not part of the module's public API)
if __name__ == "__main__":
    lp_data = LicensePlateData(text="TEST123", lpDetectionConfidence=0.88, detected=True)
    attrs = VehicleAttributes(license_plate=lp_data)
    event = VehicleEvent(
        camera_id="CAM_TEST_01",
        vehicle_track_id=101,
        base_class="car",
        confidence_base_class=0.92,
        bounding_box_frame_coords=[100, 100, 50, 30],
        attributes=attrs
    )
    print(event.model_dump_json(indent=2))
    # Example of how timestamp will be formatted:
    # "timestamp_utc": "2025-06-04T01:50:00.123456" (Pydantic v2 default)
    # If specific ISO format like "YYYY-MM-DDTHH:MM:SSZ" is strictly needed,
    # we might need a custom serializer or to format it before assignment.
    # For now, default ISO format from datetime.utcnow() is good.