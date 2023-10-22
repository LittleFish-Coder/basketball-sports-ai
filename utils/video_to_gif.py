# A program to convert a video to a gif
# - specify the starting time of the gif
# - specify the duration of the gif
# - specify the playback speed of the gif

import ffmpeg
import os

def convert_video_to_gif(input_video_path, output_gif_path, start_time=0, duration=5, playback_speed=1):
  """Converts a video to a GIF

  Args:
    input_video_path: The path to the input video file.
    output_gif_path: The path to the output GIF file.
    start_time: The starting time of the video.
    duration: The duration of the video.
    playback_speed: The playback speed. (Normal speed is 1.0)
  """

  ffmpeg.input(input_video_path, ss=start_time).filter('setpts', f'{playback_speed}*PTS').output(output_gif_path, format='gif', t=duration).run(overwrite_output=True)

if __name__ == '__main__':
  input_video_path = os.path.join(os.getcwd(), 'runs/detect/predict6/alan.avi')
  output_gif_path = os.path.join(os.getcwd(), 'src/output.gif')
  start_time = 0 # seconds
  duration = 27 # seconds
  playback_speed = 1 # the lower the value is -> the faster the video is

  convert_video_to_gif(input_video_path, output_gif_path, start_time, duration, playback_speed)