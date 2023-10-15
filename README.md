# Basketball Sports AI

NCKU Miin Wu School of Computing Project
YOLOv8 on Basketball Sports, including player detection, pose estimation.

## YOLOv8 from Ultralytics
- Official GitHub: https://github.com/ultralytics/ultralytics
- Official Doc: https://docs.ultralytics.com/
- Official Doc Usage/Configuration: https://docs.ultralytics.com/usage/cfg/ 
- Robolow(Label Platform): https://roboflow.com/

## Dataset
The dataset is provided by National Cheng Kung University Women’s Basketball Team.(Private for now)

And we use Roboflow platform to label the dataset.

## Player Action Detection
By training the model with custom dataset, we build a model to detect player actions.

The current model can detect actions below:
- Shot

![shot_detection_example](./src/object_detection_example.gif)

## Pose Estimation
By using pretrained model, we can get the pose estimation result as the video below.
![pose_estimation_example](./src/pose_estimation_example.gif)

## Future Work
- [ ] Build a player detection model
- [ ] Build a ball tracking model
- [ ] Auto clip the highlight of the game from shot detection
- [ ] Add more action detection