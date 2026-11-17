from ultralytics import YOLO
import torch

def main():
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolov8n.pt")
    model.tune(
        data="data.yml",
        epochs=2,
        patience=2,
        iterations=1,
        imgsz=640,
        batch=16,
        device=device,
        fraction=0.3,
        plots=False,
        save=False,
    )

if __name__ == "__main__":
    main()