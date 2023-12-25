from ultralytics import YOLO
import os

model = YOLO() 
pt_path = os.path.join(os.getcwd(), 'ball_rimV5.pt')
model = YOLO(pt_path) 
results = model.predict('testVideos/back.mp4',conf=0.3, classes=[0,1], show=True, max_det=1,save=False) # basketball



