from ultralytics import YOLO

def train_fire_model():
    # Load pre-trained weights from COCO
    model = YOLO("yolov8s.pt")

    # Hyperparameters tuned for 4GB VRAM (GTX 1650)
    results = model.train(
        data="datasets/data.yaml",
        epochs=50,          # Sufficient when fine-tuning from COCO weights
        patience=10,        # Stops early if mAP does not improve for 10 epochs
        batch=8,            # Prevents CUDA OOM on the 4GB RTX 3050 Laptop GPU
        imgsz=640,          # Standard resolution matching the proposal design
        device=0,           # Forces training onto the RTX 3050 GPU
        workers=4,          # Stable memory usage on 16GB DDR4 system RAM
        optimizer="AdamW",  # Stable optimizer for transfer learning
        lr0=0.001,
        amp=True,           # Automatic Mixed Precision (saves VRAM & speeds up training)
        project="runs/fire_detection",
        name="dfire_run"
    )

    # Validate against target metric (mAP@0.5 >= 0.80)
    metrics = model.val()
    print(f"Validation mAP@0.5: {metrics.box.map50:.4f}")
    print(f"Validation mAP@0.5:0.95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    train_fire_model()