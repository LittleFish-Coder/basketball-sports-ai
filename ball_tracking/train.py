from ultralytics import YOLO
import os

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
yml_path = os.path.join(os.getcwd(), 'Datasets/ball_rimV5/data.yaml')
results = model.train(data=yml_path, epochs=60, imgsz=640)
