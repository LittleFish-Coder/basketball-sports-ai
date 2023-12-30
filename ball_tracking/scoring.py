from ultralytics import YOLO
import os
import cv2
import numpy as np

video_path = os.path.join(os.getcwd(), "testing-datasets/gameplay.mp4")
ball_rim_model_path = os.path.join(os.getcwd(), "model_pt/ball_rimV8.pt")
shot_model_path = os.path.join(os.getcwd(), "model_pt/shot_detection_v2.pt")

ball_rim_model = YOLO(ball_rim_model_path)
shot_model = YOLO(shot_model_path)

# load video
cap = cv2.VideoCapture(video_path)

# get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


# write out video
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

rim_bounding_box = None  # xyxy: [x1, y1, x2, y2] from top left to bottom right
rim_location = None  # xywh: [x, y, w, h] center of the box, width, height
ball_location = None  # xywh: [x, y, w, h] center of the box, width, height
previous_ball_location = None  # xywh: [x, y, w, h] center of the box, width, height
ball_tracking = []  # list of ball location
ball_tracking_history = []  # list of ball tracking
shot_detected = False  # flag to indicate if the shot is detected
frame_to_count = fps * 2  # set how many frames to count to avoid keep tracking the ball after the shot


def intersect(line1: list[tuple], line2: list[tuple]):
    [(x1, y1), (x2, y2)] = line1
    [(x3, y3), (x4, y4)] = line2

    # Calculate slopes (m) and y-intercepts (b) for the two lines
    m1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float("inf")  # Avoid division by zero
    b1 = y1 - m1 * x1 if m1 != float("inf") else None

    m2 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else float("inf")  # Avoid division by zero
    b2 = y3 - m2 * x3 if m2 != float("inf") else None

    # Check for parallel lines
    if m1 == m2:
        # Check for overlapping lines
        if b1 == b2:
            return True  # Lines overlap
        else:
            return False  # Lines are parallel but not overlapping
    else:
        # Check for intersection point
        x_intersect = (b2 - b1) / (m1 - m2) if m1 != float("inf") else x1
        y_intersect = m1 * x_intersect + b1 if m1 != float("inf") else m2 * x_intersect + b2

        # Check if the intersection point lies within the line segments
        if (
            min(x1, x2) <= x_intersect <= max(x1, x2)
            and min(y1, y2) <= y_intersect <= max(y1, y2)
            and min(x3, x4) <= x_intersect <= max(x3, x4)
            and min(y3, y4) <= y_intersect <= max(y3, y4)
        ):
            return True  # Lines intersect
        else:
            return False  # Lines are not parallel but do not intersect


# detect rim first
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # run yolo inference on the frame, classes= {0: "ball", 1: "rim"}
        rim = ball_rim_model.predict(frame, classes=[1], max_det=1)

        if rim[0].boxes.__len__() != 0:
            rim_bounding_box = rim[0].boxes.data[0].numpy().astype(int)  # xyxy: [x1, y1, x2, y2] from top left to bottom right
            rim_location = rim[0].boxes.xywh[0].numpy().astype(int)  # xywh: [x, y, w, h] center of the box, width, height
            # counvert it to dictionary
            rim_location = {"x": rim_location[0], "y": rim_location[1], "w": rim_location[2], "h": rim_location[3]}
            print(f"get rim_location: {rim_location}")
            # draw the rim location on the black image
            # black_image = np.zeros((height, width, 3), np.uint8)
            # cv2.rectangle(black_image, (rim_bounding_box[0], rim_bounding_box[1]), (rim_bounding_box[2], rim_bounding_box[3]), (255, 255, 255), 2)
            # cv2.imshow("rim_location", black_image)
            # cv2.waitKey(0)
            break

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# standard line of deiciding whether the ball is in or out
standard_line = [(rim_location["x"] - rim_location["w"] // 2, rim_location["y"]), (rim_location["x"] + rim_location["w"] // 2, rim_location["y"])]

# color to draw the ball tracking (B, G, R)
color = tuple(np.random.randint(0, 255, size=(1, 3), dtype="uint8").squeeze().tolist())

# reset the video capture object
cap = cv2.VideoCapture(video_path)

# loop through the video frames
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()

    if success:
        if shot_detected:
            frame_count += 1

        # run yolo inference on the frame, classes= {0: "ball", 1: "rim"}
        ball = ball_rim_model.predict(frame, classes=[0], max_det=1, conf=0.5)
        shot = shot_model.predict(frame, max_det=1, conf=0.5)

        # check if results contains shot
        if shot[0].boxes.__len__() != 0:
            shot_detected = True
            cv2.putText(frame, "Shot Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # check if results contains ball
        if ball[0].boxes.__len__() != 0:
            ball_location = ball[0].boxes.xywh[0].numpy().astype(int)  # xywh: [x, y, w, h] center of the box, width, height
            # print(f"get ball_location: {ball_location}")
            if shot_detected:
                ball_tracking.append(ball_location)

        # plot the history of the ball tracking
        if len(ball_tracking_history) > 0:
            for history in ball_tracking_history:
                for i in range(len(history["ball_tracking"]) - 1):
                    cv2.line(frame, tuple(history["ball_tracking"][i][:2]), tuple(history["ball_tracking"][i + 1][:2]), history["color"], 2)

        # track current the ball
        if len(ball_tracking) > 1 and frame_count <= frame_to_count:
            # draw the ball tracking
            for i in range(len(ball_tracking) - 1):
                cv2.line(frame, tuple(ball_tracking[i][:2]), tuple(ball_tracking[i + 1][:2]), color, 2)
        elif frame_count > frame_to_count:
            # clear the ball tracking
            ball_tracking = []
            shot_detected = False
            frame_count = 0

        # trajectory of the previous ball and current ball location
        trajectory = None
        if previous_ball_location is not None and ball_location is not None:
            trajectory = [tuple(previous_ball_location[:2]), tuple(ball_location[:2])]
            cv2.line(frame, tuple(previous_ball_location[:2]), tuple(ball_location[:2]), (255, 0, 0), 2)

        # draw the standard line
        cv2.line(frame, standard_line[0], standard_line[1], (0, 255, 0), 2)
        # plot the rim bounding box with rectangle
        cv2.rectangle(frame, (rim_bounding_box[0], rim_bounding_box[1]), (rim_bounding_box[2], rim_bounding_box[3]), (0, 255, 0), 2)

        # check if the ball is in the rim
        if trajectory is not None:
            if intersect(trajectory, standard_line):
                cv2.putText(frame, "In", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # save the ball tracking
                history = {
                    "ball_tracking": ball_tracking,
                    "color": color,
                }
                ball_tracking_history.append(history)
                # clear the ball tracking
                frame_count = 0
                ball_tracking = []
                shot_detected = False
                # reset the color
                color = tuple(np.random.randint(0, 255, size=(1, 3), dtype="uint8").squeeze().tolist())

        # visualize the results on the frame
        # args: conf=False, labels=False, boxes=False
        annotated_frame = ball[0].plot(conf=False, labels=False, boxes=False)  # ball bounding box
        # annotated_frame = shot[0].plot(conf=False)  # shot bounding box

        # write out video
        out.write(annotated_frame)
        # display the annotated frame
        cv2.imshow("Ball Tracking", annotated_frame)

        # update the previous ball location
        if ball_location is not None:
            previous_ball_location = ball_location
            ball_location = None

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and video writer object
cap.release()
out.release()
cv2.destroyAllWindows()
