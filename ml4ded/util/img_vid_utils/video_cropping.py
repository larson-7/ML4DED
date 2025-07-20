import subprocess
import re
import os

def detect_crop_params(video_path, sample_duration=5):
    """Use FFmpeg cropdetect to get optimal crop parameters from the first few seconds of the video."""
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"cropdetect=24:16:0",
        "-t", str(sample_duration),
        "-f", "null", "-"
    ]

    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)

    if not matches:
        raise RuntimeError("Could not detect crop parameters. Check your video.")

    # Use the last detected crop (most stable estimate)
    crop_param = matches[-1]
    print(f"Detected crop: {crop_param}")
    return crop_param

def crop_video_ffmpeg(input_path, output_path, crop_param):
    """Crop the video using FFmpeg with the given crop parameters."""
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"crop={crop_param}",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    print(f"Cropped video saved to: {output_path}")

if __name__ == "__main__":
    input_video = "/home/jordan/omscs/ML4DED/data/DEDWallVideos/buildplate000_5.mp4"
    output_dir = "/home/jordan/omscs/ML4DED/data/DEDWallVideos_Cropped"
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, "buildplate000_5.mp4")
    # Step 1: Detect crop parameters
    crop = detect_crop_params(input_video)

    # Step 2: Apply cropping
    crop_video_ffmpeg(input_video, output_video, crop)
