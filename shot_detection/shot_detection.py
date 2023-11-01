from ultralytics import YOLO
import os
import cv2

# Load model
pt = os.path.join(os.getcwd(), "model_pt/shot_detection_v2.pt")
model = YOLO(pt)

# Inference
source = os.path.join(os.getcwd(), "testing-datasets/alan_stadium.mp4")
results = model(source, save=True, conf=0.3, show_labels=True, boxes=True, stream=False)

# Open the video file
video_path = os.path.join(os.getcwd(), "testing-datasets/alan_stadium.mp4")
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
