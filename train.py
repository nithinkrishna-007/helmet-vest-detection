from ultralytics import YOLO
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yml",
        epochs=100,
        patience=2,     # early stopping: stop after 2 epochs with no improvement
        imgsz=640,
        batch=16,
        device=0,        # use the first GPU
    )

if __name__ == "__main__":
    main()