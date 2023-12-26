from ultralytics import YOLO
import os
import cv2

video_path = os.path.join(os.getcwd(), "testing-datasets/side.mp4")
model_path = os.path.join(os.getcwd(), "model_pt/ball_rimV5.pt")

model = YOLO(model_path)

# load video
cap = cv2.VideoCapture(video_path)

# get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# write out video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # run yolo inference on the frame
        results = model(frame)

        print(results[0].boxes)

        break

        # visualize the results on the frame
        annotated_frame = results[0].plot()

        # write out video
        out.write(annotated_frame)
        # display the annotated frame
        cv2.imshow("YOLOv5 Inference", annotated_frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

    # Release the video capture object and video writer object
    cap.release()
    out.release()
    cv2.destroyAllWindows()
