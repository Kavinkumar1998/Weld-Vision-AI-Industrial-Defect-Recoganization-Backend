from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime

class WeldVerdictEnum(str, Enum):
    GOOD = "GOOD"
    DEFECTIVE = "DEFECTIVE"
    NO_WELD_DETECTED = "NO_WELD_DETECTED"
    UNKNOWN = "UNKNOWN"

class BoundingBoxSchema(BaseModel):
    x_min: float = Field(..., description="Top-left X pixel coordinate")
    y_min: float = Field(..., description="Top-left Y pixel coordinate")
    x_max: float = Field(..., description="Bottom-right X pixel coordinate")
    y_max: float = Field(..., description="Bottom-right Y pixel coordinate")

class DetectedDefectSchema(BaseModel):
    id: str = Field(..., description="Unique detection identifier")
    class_id: int = Field(..., description="YOLO Class ID (0, 1, 2)")
    class_name: str = Field(..., description="Class name ('Bad Weld', 'Good Weld', 'Defect')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    box: BoundingBoxSchema
    is_defective: bool = Field(..., description="True if classification indicates a defect")
    color_hex: str = Field(default="#EF4444", description="Hex display color")

class ImageDetectionResponse(BaseModel):
    success: bool = True
    verdict: WeldVerdictEnum
    verdict_confidence: float = Field(..., ge=0.0, le=1.0)
    total_detections: int
    defect_count: int
    good_weld_count: int
    processing_time_ms: float
    detections: List[DetectedDefectSchema]
    annotated_image_base64: Optional[str] = None
    image_width: int
    image_height: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StreamFrameResponse(BaseModel):
    type: str = "FRAME_RESULT"
    verdict: WeldVerdictEnum
    verdict_confidence: float
    total_detections: int
    defect_count: int
    good_weld_count: int
    latency_ms: float
    fps: float
    detections: List[DetectedDefectSchema]
    frame_timestamp: Optional[float] = None

class ModelInfoResponse(BaseModel):
    model_name: str
    classes: dict
    weights_path: str
    device: str
    is_loaded: bool
