# convert video format from .mkv to .mp4
import ffmpeg
import os

input_video_path = os.path.join(os.getcwd(), 'runs/detect/predict8/alan.avi')
output_video_path = os.path.join(os.getcwd(), 'src/alan.mp4')

# convert video format
ffmpeg.input(input_video_path).output(output_video_path).run()