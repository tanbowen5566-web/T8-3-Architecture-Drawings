from pathlib import Path
from ultralytics import YOLO

# Path to the pretrained YOLO model
MODEL_PATH = Path(__file__).parent / "drawing_checker.pt"


def detect_views(image_path):
    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=0.6,
        imgsz=640,
        verbose=False
    )

    views = []

    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        views.append({
            "view_id": i + 1,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(box.conf[0])
        })

    return views
