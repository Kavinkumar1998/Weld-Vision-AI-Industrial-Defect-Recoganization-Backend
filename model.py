import os
import time
import uuid
import base64
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from backend.schemas import (
        BoundingBoxSchema,
        DetectedDefectSchema,
        WeldVerdictEnum
    )
except ImportError:
    from schemas import (
        BoundingBoxSchema,
        DetectedDefectSchema,
        WeldVerdictEnum
    )

logger = logging.getLogger("weldvision.model")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

class YOLOModel:
    """Wrapper around Ultralytics YOLO model for weld flaw detection and classification."""

    def __init__(self, weights_path: Optional[str] = None):
        self.base_dir = Path(__file__).resolve().parent
        
        # Priority order for weights
        candidate_paths = [
            Path(weights_path) if weights_path else None,
            self.base_dir / "weights" / "best.pt",
            self.base_dir.parent / "src" / "runs" / "detect" / "weld_detection_project" / "weld_yolo_training" / "weights" / "best.pt",
            self.base_dir.parent / "models" / "weights" / "best.pt"
        ]

        self.weights_path: str = ""
        for path in candidate_paths:
            if path and path.exists():
                self.weights_path = str(path)
                break

        self.model: Optional[YOLO] = None
        self.class_names: Dict[int, str] = {
            0: "Bad Weld",
            1: "Good Weld",
            2: "Defect"
        }
        self.good_class_ids = {1}
        self.defective_class_ids = {0, 2}
        self.is_loaded: bool = False

    def load(self, device: str = "cpu") -> None:
        """Load model weights and perform kernel warmup."""
        if not self.weights_path:
            raise FileNotFoundError("Could not locate best.pt weights file.")

        logger.info(f"Loading YOLO model weights from: {self.weights_path}")
        self.model = YOLO(self.weights_path)

        if hasattr(self.model, "names") and self.model.names:
            self.class_names = self.model.names
            logger.info(f"Model class mapping: {self.class_names}")

        self._warmup(device)
        self.is_loaded = True
        logger.info("YOLO Model is ready for inference.")

    def _warmup(self, device: str) -> None:
        """Warm up CUDA/CPU kernels with synthetic dummy forward pass."""
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(source=dummy, conf=0.25, device=device, verbose=False)
            logger.info("Model warmup completed successfully.")
        except Exception as e:
            logger.warning(f"Model warmup notice: {e}")

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.45,
        device: str = "cpu"
    ) -> Tuple[WeldVerdictEnum, float, List[DetectedDefectSchema], float]:
        """
        Run inference on a BGR image array.
        Returns: (verdict, verdict_confidence, detections, latency_ms)
        """
        if not self.is_loaded or self.model is None:
            self.load(device=device)

        start = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            verbose=False
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        detections: List[DetectedDefectSchema] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"Class {cls_id}")

                is_defective = cls_id in self.defective_class_ids
                color_hex = "#EF4444" if is_defective else "#10B981"

                item = DetectedDefectSchema(
                    id=str(uuid.uuid4())[:8],
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=round(conf, 4),
                    box=BoundingBoxSchema(
                        x_min=round(float(xyxy[0]), 2),
                        y_min=round(float(xyxy[1]), 2),
                        x_max=round(float(xyxy[2]), 2),
                        y_max=round(float(xyxy[3]), 2),
                    ),
                    is_defective=is_defective,
                    color_hex=color_hex
                )
                detections.append(item)

        # Compute weld classification verdict
        defects = [d for d in detections if d.is_defective]
        goods = [d for d in detections if not d.is_defective]

        if defects:
            best_defect = max(defects, key=lambda d: d.confidence)
            verdict = WeldVerdictEnum.DEFECTIVE
            verdict_conf = best_defect.confidence
        elif goods:
            best_good = max(goods, key=lambda d: d.confidence)
            verdict = WeldVerdictEnum.GOOD
            verdict_conf = best_good.confidence
        else:
            verdict = WeldVerdictEnum.NO_WELD_DETECTED
            verdict_conf = 0.0

        return verdict, verdict_conf, detections, latency_ms

    @staticmethod
    def decode_image(image_bytes: bytes) -> Optional[np.ndarray]:
        """Decode raw image bytes into an OpenCV BGR numpy array in memory (Zero disk I/O)."""
        if not image_bytes:
            return None
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def decode_base64(base64_str: str) -> Optional[np.ndarray]:
        """Decode a Base64 encoded image string."""
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
            raw_bytes = base64.b64decode(base64_str)
            return YOLOModel.decode_image(raw_bytes)
        except Exception:
            return None

    @staticmethod
    def encode_base64(image: np.ndarray, quality: int = 85) -> str:
        """Encode an OpenCV image to JPEG in memory and return as base64 data URI."""
        success, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            raise ValueError("Failed to encode image to JPEG buffer.")
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    @staticmethod
    def annotate(image: np.ndarray, detections: List[DetectedDefectSchema], verdict: WeldVerdictEnum) -> np.ndarray:
        """Draw bounding boxes and status banner onto an image copy."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        for det in detections:
            x1 = int(max(0, min(w - 1, det.box.x_min)))
            y1 = int(max(0, min(h - 1, det.box.y_min)))
            x2 = int(max(0, min(w - 1, det.box.x_max)))
            y2 = int(max(0, min(h - 1, det.box.y_max)))

            color = (34, 197, 94) if not det.is_defective else (59, 59, 239) # BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            label = f"{det.class_name} {det.confidence * 100:.1f}%"
            (font_w, font_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            
            label_y1 = max(0, y1 - font_h - 10)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + font_w + 10, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Header Verdict Banner
        banner_h = max(45, int(h * 0.07))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (30, 41, 59), -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        if verdict == WeldVerdictEnum.DEFECTIVE:
            v_text = "VERDICT: DEFECTIVE WELD"
            v_color = (59, 59, 239)
        elif verdict == WeldVerdictEnum.GOOD:
            v_text = "VERDICT: GOOD WELD (PASS)"
            v_color = (34, 197, 94)
        elif verdict == WeldVerdictEnum.NO_WELD_DETECTED:
            v_text = "VERDICT: NO WELD DETECTED"
            v_color = (180, 180, 180)
        else:
            v_text = "VERDICT: UNKNOWN"
            v_color = (200, 200, 200)

        cv2.putText(annotated, v_text, (16, int(banner_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, w / 1000.0), v_color, 2, cv2.LINE_AA)
        return annotated

# Global model instance
model = YOLOModel()
