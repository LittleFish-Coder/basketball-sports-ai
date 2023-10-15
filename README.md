# Basketball Sports AI

NCKU Miin Wu School of Computing Project
YOLOv8 on Basketball Sports, including player detection, pose estimation.

## YOLOv8 from Ultralytics
- Official GitHub: https://github.com/ultralytics/ultralytics
- Official Doc: https://docs.ultralytics.com/
- Official Doc Usage/Configuration: https://docs.ultralytics.com/usage/cfg/ 
- Robolow(Label Platform): https://roboflow.com/

## Player Action Detection
The data set is provided by National Cheng Kung University Women’s Basketball Team.
And we use Roboflow platform to label the data set.

The current classes are as follows:
- Shot

By using the online model from Roboflow, we can get the player action detection result as the video below.
![shot](./src/object_detection_example.gif)

## Pose Estimation
By using pretrained model, we can get the pose estimation result as the video below.
![example](./src/pose_estimation_example.gif)

with argument config:
- conf: Confidence threshold for object detection
- show_labels: Show labels in object detection predictions
- show_conf: Show confidence scores in object detection predictions
- max_det: Maximum number of detections per image
- boxes: Show boxes in segmentation predictions

We can get the result as the video below.
![example2](./src/pose_estimation_example2.gif)
