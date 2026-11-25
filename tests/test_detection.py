import io
import os
import cv2
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from backend.model import YOLOModel

client = TestClient(app)

def create_synthetic_image_bytes():
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 320), 100, (200, 200, 200), -1)
    success, buf = cv2.imencode(".jpg", img)
    assert success
    return io.BytesIO(buf.tobytes())

def test_history_thumbnail_dimensions_are_bounded():
    image = np.zeros((1200, 1600, 3), dtype=np.uint8)

    thumbnail = YOLOModel.encode_thumbnail(image)
    decoded = YOLOModel.decode_base64(thumbnail)

    assert decoded is not None
    assert max(decoded.shape[:2]) <= 320

def test_detect_image_endpoint():
    img_bytes = create_synthetic_image_bytes()
    response = client.post(
        "/detect/image?confidence=0.25",
        files={"file": ("synthetic_weld.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "verdict" in data
    assert "detections" in data
    assert "processing_time_ms" in data
    assert data["image_width"] == 640
    assert data["image_height"] == 640

def test_detect_image_with_existing_test_dataset():
    sample_path = "src/datasets/test/images/4a05306b-ca866e1f-c12_jpg.rf.c5c926823394e6056b533bedd6e64c99.jpg"
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            response = client.post(
                "/api/v1/detect/image",
                files={"file": ("weld_test.jpg", f, "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["verdict"] in ["GOOD", "DEFECTIVE", "NO_WELD_DETECTED"]
