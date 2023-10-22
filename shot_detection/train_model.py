from ultralytics import YOLO
import os

# # load pretrained model
pt = os.path.join(os.getcwd(), 'model_pt/yolov8n.pt')
model = YOLO(pt)  # You can also specify `yolov8m.pt` or other pretrained weights

# train the model on custom dataset
yml_path = os.path.join(os.getcwd(), 'datasets/Shot Detection Version 2 Dataset/data.yaml')
results = model.train(data=yml_path, epochs=10)  # the result will be saved in runs/detect/train/weights