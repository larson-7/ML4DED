import subprocess
import re
import os
import cv2

def detect_crop_params(video_path, sample_duration=5):
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", "cropdetect=24:16:0",
        "-t", str(sample_duration),
        "-f", "null", "-"
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    matches = re.findall(r"crop=(\d+:\d+:\d+:\d+)", result.stderr)
    if not matches:
        raise RuntimeError("Could not detect crop parameters. Check your video.")
    crop_param = matches[-1]
    print(f"Detected crop: {crop_param}")
    return crop_param

def crop_video_ffmpeg(input_path, output_path, crop_param):
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"crop={crop_param}",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(command, check=True)
    print(f"Cropped video saved to: {output_path}")

def extract_frames(video_path, frame_output_dir, stride=5):
    os.makedirs(frame_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: Cannot open cropped video for frame extraction.")
    frame_idx = 0
    saved_paths = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            frame_path = os.path.join(frame_output_dir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
        frame_idx += 1
    cap.release()
    print(f"Extracted {len(saved_paths)} frames to {frame_output_dir}")
    return saved_paths

def crop_and_save_video(input_video, output_dir, sample_duration=5, stride=5):
    """
    Detects crop parameters, crops the video, extracts frames, and returns paths.
    Returns: (cropped_video_path, [frame_paths])
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(input_video)
    cropped_video_path = os.path.join(output_dir, video_name)

    # Crop detection and crop
    crop = detect_crop_params(input_video, sample_duration=sample_duration)
    crop_video_ffmpeg(input_video, cropped_video_path, crop)

    # Extract frames from cropped video
    video_stem = os.path.splitext(video_name)[0]
    frame_output_dir = os.path.join(output_dir, f"{video_stem}_frames")
    frame_paths = extract_frames(cropped_video_path, frame_output_dir, stride=stride)

    return cropped_video_path, frame_paths

if __name__ == "__main__":
    input_video = "/home/jordan/omscs/ML4DED/data/DEDWallVideos/buildplate000_5.mp4"
    output_dir = "/home/jordan/omscs/ML4DED/data/DEDWallVideos_Cropped"
    cropped_video_path, frame_paths = crop_and_save_video(input_video, output_dir)
    print("Cropped video:", cropped_video_path)
    print("Frames:", frame_paths[:5], "...")  # Show just the first few
