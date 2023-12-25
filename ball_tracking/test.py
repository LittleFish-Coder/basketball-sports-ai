from ultralytics import YOLO
import os

pt_path = os.path.join(os.getcwd(), "model_pt/ball_rimV5.pt")
model = YOLO(pt_path)

# show how many class in the model
print(model.names)

# inference
results = model.predict("testing-datasets/side.mp4", conf=0.3, save=True)  # basketball
