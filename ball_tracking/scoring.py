from ultralytics import YOLO
import os
import cv2
import numpy as np

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

rim_location = None
ball_location = None
ball_tracking = []

# detect rim first
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # run yolo inference on the frame, classes= {0: "ball", 1: "rim"}
        results = model.predict(frame, classes=[1], max_det=1)

        if results[0].boxes.data is not None:
            rim_location = results[0].boxes.data[0].numpy().astype(int)
            print(f"get rim_location: {rim_location}")
            # draw the rim location on the black image
            # black_image = np.zeros((height, width, 3), np.uint8)
            # cv2.rectangle(black_image, (rim_location[0], rim_location[1]), (rim_location[2], rim_location[3]), (255, 255, 255), 2)
            # cv2.imshow("rim_location", black_image)
            # cv2.waitKey(0)
            break

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # run yolo inference on the frame, classes= {0: "ball", 1: "rim"}
        results = model.predict(frame, classes=[0], max_det=1)

        # get the ball location
        if results[0].boxes.__len__() != 0:  # check if results contains ball
            ball_location = results[0].boxes.data[0].numpy().astype(int)
            print(f"get ball_location: {ball_location}")
            ball_tracking.append(ball_location)

        # track the ball
        if len(ball_tracking) > 1:
            # draw the ball tracking
            for i in range(len(ball_tracking) - 1):
                cv2.line(frame, tuple(ball_tracking[i][:2]), tuple(ball_tracking[i + 1][:2]), (0, 0, 255), 2)

            # calculate the distance between the ball location and the rim location
            distance = np.linalg.norm(ball_location - rim_location)
            print(f"distance: {distance}")
            # if the distance is less than 50, then the ball is in the rim
            if distance < 50:
                print("ball in the rim")

                # write the word "ball in the rim" on the frame
                cv2.putText(frame, "ball in the rim", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # visualize the results on the frame
        annotated_frame = results[0].plot()

        # write out video
        out.write(annotated_frame)
        # display the annotated frame
        cv2.imshow("Ball Tracking", annotated_frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and video writer object
cap.release()
# out.release()
cv2.destroyAllWindows()
