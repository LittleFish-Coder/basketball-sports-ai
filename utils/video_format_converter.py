# convert video format from .mkv to .mp4
import argparse
import ffmpeg

def convert_video(input_path, output_path):
    ffmpeg.input(input_path).output(output_path).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Format Converter")
    parser.add_argument("--input", default="./runs/detect/predict8/alan.avi", help="Input video path (e.g., /path/to/video.mkv)")
    parser.add_argument("--output", default="./src/alan.mp4", help="Output video path (e.g., /path/to/output.mp4)")

    args = parser.parse_args()

    input_video_path = args.input
    output_video_path = args.output

    convert_video(input_video_path, output_video_path)
