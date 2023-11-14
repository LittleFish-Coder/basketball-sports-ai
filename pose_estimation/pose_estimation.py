from ultralytics import YOLO
import os
import cv2
import numpy as np
from utils import calculate_degree


def calculate_degree(pt1: tuple, pt2: tuple, pt3: tuple):
    """
    Calculate the angle between three points
    """
    # Convert tuple to numpy array
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pt3 = np.array(pt3)

    # Calculate vectors
    vec1 = pt2 - pt1
    vec2 = pt3 - pt2

    # Calculate dot product
    dot_product = np.dot(vec1, vec2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    # Calculate angle
    angle = np.arccos(dot_product / (magnitude1 * magnitude2))

    # Convert to degrees
    degrees = np.degrees(angle)

    # Calculate the interior angle
    interior_angle = 180 - degrees

    return interior_angle.round(2)


# Load pretrained model
pt_path = os.path.join(os.getcwd(), "model_pt/yolov8n-pose.pt")
model = YOLO(pt_path)

# Load image
img_path = os.path.join(os.getcwd(), "testing-datasets/human.jpg")

# Inference
results = model(img_path)

# Get the keypoints
for r in results:
    print(r.keypoints.xy[0])
    keypoints = r.keypoints.xy[0].numpy().astype(int)
    # get left shoulder
    print(keypoints[5])
    # get left elbow
    print(keypoints[7])
    # get left wrist
    print(keypoints[9])

    # calculate the angle degree
    degree = calculate_degree(tuple(keypoints[5]), tuple(keypoints[7]), tuple(keypoints[9]))
    print(degree)

    # draw the line
    img = cv2.imread(img_path)
    img = cv2.line(img, tuple(keypoints[5]), tuple(keypoints[7]), (0, 255, 0), 3)
    img = cv2.line(img, tuple(keypoints[7]), tuple(keypoints[9]), (255, 0, 0), 3)

    # show the angle
    cv2.putText(img, f"{degree}", tuple(keypoints[7]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
