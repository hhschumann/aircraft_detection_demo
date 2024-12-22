from ultralytics import YOLO

model_path = "./medium/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

model.export(format="onnx") 