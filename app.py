import uvicorn
import os
import sys

# Ensure root and current directory are in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
for p in [project_root, current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

if __name__ == "__main__":
    print("=" * 60)
    print("  WeldVision AI — Industrial Defect Detection")
    print("=" * 60)
    print("  Backend API:      http://localhost:8000")
    print("  Swagger Docs:     http://localhost:8000/docs")
    print("  WebSocket Stream: ws://localhost:8000/api/v1/detect/stream")
    print("=" * 60)
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
