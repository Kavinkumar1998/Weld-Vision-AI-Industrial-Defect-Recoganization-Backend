import os
import sys
import time
import json
import asyncio
from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, Deque

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [PROJECT_ROOT, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from backend.model import model
    from backend.schemas import (
        ImageDetectionResponse,
        StreamFrameResponse,
        ModelInfoResponse,
        WeldVerdictEnum
    )
except ImportError:
    from model import model
    from schemas import (
        ImageDetectionResponse,
        StreamFrameResponse,
        ModelInfoResponse,
        WeldVerdictEnum
    )

START_TIME = time.time()
INSPECTION_HISTORY: Deque[dict] = deque(maxlen=100)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load YOLO model weights into memory
    print("=" * 60)
    print("  Starting WeldVision AI Backend Server...")
    print("=" * 60)
    try:
        model.load(device="cpu")
        print("  YOLO Model Loaded Successfully!")
    except Exception as e:
        print(f"  Warning: Model loading error at startup: {e}")
    yield
    print("  Stopping WeldVision AI Backend Server.")

app = FastAPI(
    title="WeldVision AI — Model Backend",
    version="1.0.0",
    description="Asynchronous YOLOv8 Weld Defect Detection & Real-Time Streaming API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== HEALTH & METADATA ====================

@app.get("/health", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    import torch
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    return {
        "status": "healthy",
        "service": "WeldVision AI Backend",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "model_loaded": model.is_loaded,
        "device": device_name,
        "cuda_available": cuda_available,
        "classes": model.class_names
    }

@app.get("/model/info", tags=["Health"])
@app.get("/api/v1/model/info", tags=["Health"])
async def model_info():
    return {
        "model_name": "YOLOv8 Custom Weld Defect Detector",
        "weights_path": model.weights_path,
        "is_loaded": model.is_loaded,
        "classes": model.class_names,
        "defect_classes": list(model.defective_class_ids),
        "good_classes": list(model.good_class_ids)
    }

# ==================== IMAGE INFERENCE ====================

@app.post("/detect/image", response_model=ImageDetectionResponse, tags=["Inference"])
@app.post("/api/v1/detect/image", response_model=ImageDetectionResponse, tags=["Inference"])
async def detect_image(
    file: UploadFile = File(..., description="Multipart image file"),
    confidence: float = Query(default=0.40, ge=0.05, le=1.0),
    iou_threshold: float = Query(default=0.45, ge=0.1, le=0.9),
    return_annotated_image: bool = Query(default=True)
):
    # 1. Read bytes in-memory (Zero disk I/O)
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 20MB limit.")

    # 2. Decode in-memory
    image = model.decode_image(contents)
    if image is None:
        raise HTTPException(status_code=400, detail="Corrupted or unreadable image file.")

    h, w = image.shape[:2]

    # 3. Offload YOLO prediction to thread pool (Async unblocked)
    try:
        verdict, verdict_conf, detections, latency_ms = await asyncio.to_thread(
            model.predict,
            image=image,
            conf_threshold=confidence,
            iou_threshold=iou_threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

    # 4. Optional in-memory annotation
    annotated_b64: Optional[str] = None
    if return_annotated_image:
        try:
            annotated_img = model.annotate(image, detections, verdict)
            annotated_b64 = model.encode_base64(annotated_img)
        except Exception as e:
            pass

    defect_count = sum(1 for d in detections if d.is_defective)
    good_weld_count = sum(1 for d in detections if not d.is_defective)

    response = ImageDetectionResponse(
        success=True,
        verdict=verdict,
        verdict_confidence=round(verdict_conf, 4),
        total_detections=len(detections),
        defect_count=defect_count,
        good_weld_count=good_weld_count,
        processing_time_ms=round(latency_ms, 2),
        detections=detections,
        annotated_image_base64=annotated_b64,
        image_width=w,
        image_height=h,
        timestamp=datetime.utcnow()
    )

    # 5. History audit log
    INSPECTION_HISTORY.appendleft({
        "id": f"WELD-{int(datetime.utcnow().timestamp() * 1000)}",
        "timestamp": response.timestamp.isoformat(),
        "filename": file.filename or "upload.jpg",
        "verdict": response.verdict.value,
        "verdict_confidence": response.verdict_confidence,
        "total_detections": response.total_detections,
        "defect_count": response.defect_count,
        "good_weld_count": response.good_weld_count,
        "latency_ms": response.processing_time_ms,
        "thumbnail_b64": annotated_b64
    })

    return response

# ==================== WEBSOCKET STREAMING ====================

@app.websocket("/detect/stream")
@app.websocket("/api/v1/detect/stream")
async def websocket_detect_stream(websocket: WebSocket):
    await websocket.accept()
    confidence = 0.40
    iou_threshold = 0.45
    is_busy = False
    last_frame_time = time.perf_counter()

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            # Frame dropping gate: if worker is processing previous frame, skip incoming frame
            if is_busy:
                continue

            raw_bytes: Optional[bytes] = None
            frame_ts: Optional[float] = None

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("event") == "CONFIG":
                        confidence = float(payload.get("confidence", confidence))
                        iou_threshold = float(payload.get("iou", iou_threshold))
                        await websocket.send_json({"type": "CONFIG_ACK", "confidence": confidence, "iou": iou_threshold})
                        continue
                    if "frame" in payload:
                        raw_bytes = payload["frame"]
                        frame_ts = payload.get("timestamp")
                except json.JSONDecodeError:
                    continue

            if not raw_bytes:
                continue

            is_busy = True
            try:
                if isinstance(raw_bytes, str):
                    image = model.decode_base64(raw_bytes)
                else:
                    image = model.decode_image(raw_bytes)

                if image is None:
                    continue

                # Run inference asynchronously
                verdict, verdict_conf, detections, latency_ms = await asyncio.to_thread(
                    model.predict,
                    image=image,
                    conf_threshold=confidence,
                    iou_threshold=iou_threshold
                )

                now = time.perf_counter()
                fps = round(1.0 / max(0.001, (now - last_frame_time)), 1)
                last_frame_time = now

                defect_count = sum(1 for d in detections if d.is_defective)
                good_weld_count = sum(1 for d in detections if not d.is_defective)

                response = StreamFrameResponse(
                    type="FRAME_RESULT",
                    verdict=verdict,
                    verdict_confidence=round(verdict_conf, 4),
                    total_detections=len(detections),
                    defect_count=defect_count,
                    good_weld_count=good_weld_count,
                    latency_ms=round(latency_ms, 1),
                    fps=fps,
                    detections=detections,
                    frame_timestamp=frame_ts
                )

                await websocket.send_text(response.model_dump_json())
            finally:
                is_busy = False

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket stream error: {e}")

# ==================== HISTORY AUDIT ====================

@app.get("/history", tags=["History"])
@app.get("/api/v1/history", tags=["History"])
async def get_history(limit: int = Query(default=30, ge=1, le=100)):
    return {
        "total_cached": len(INSPECTION_HISTORY),
        "records": list(INSPECTION_HISTORY)[:limit]
    }

@app.delete("/history", tags=["History"])
@app.delete("/api/v1/history", tags=["History"])
async def clear_history():
    INSPECTION_HISTORY.clear()
    return {"status": "cleared"}

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "WeldVision AI — Model Backend",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "detect_image": "POST /detect/image (or /api/v1/detect/image)",
            "detect_stream": "WS /detect/stream (or /api/v1/detect/stream)",
            "history": "GET /history (or /api/v1/history)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False
    )
