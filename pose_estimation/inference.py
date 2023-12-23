from ultralytics import YOLO
import os
import cv2
import numpy as np
from utils import calculate_degree

video_path = os.path.join(os.getcwd(), "testing-datasets/side.mp4")
model_path = os.path.join(os.getcwd(), "model_pt/yolov8n-pose.pt")


def main():
    # Load pretrained model
    model = YOLO(model_path)

    # Load video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # write out video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    # Loop through the video frames
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, max_det=1)

            # Get the keypoints
            keypoints = results[0].keypoints.xy[0].numpy().astype(int)
            # Calculate the elbow angle degree: right shoulder, right elbow, right wrist
            elbow_degree = calculate_degree(keypoints[4], keypoints[6], keypoints[8])
            # Calculate the knee angle degree: right hips, right knee, right ankle
            knee_degree = calculate_degree(keypoints[12], keypoints[14], keypoints[16])

            # Visualize the results on the frame
            annotated_frame = results[0].plot(boxes=False, kpt_radius=1)
            # Draw the elbow angle degree
            cv2.putText(annotated_frame, f"{elbow_degree}", tuple(keypoints[6] + [25, 25]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # Draw the knee angle degree
            cv2.putText(annotated_frame, f"{knee_degree}", tuple(keypoints[14] + [5, 5]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Write out video
            out.write(annotated_frame)
            # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
    # Release the video capture object and video writer object
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
