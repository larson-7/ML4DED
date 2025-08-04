def process_videos(data_dir):
    import os
    import cv2
    from video_cropping import detect_crop_params, crop_video_ffmpeg

    cropped_dir = os.path.join(data_dir, "cropped")
    os.makedirs(cropped_dir, exist_ok=True)

    frame_dir = os.path.join(data_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    videos = [item for item in os.listdir(data_dir) if item.endswith(".mp4") and os.path.isfile(os.path.join(data_dir, item))]
    print(f"Found {len(videos)} videos")

    for video in videos:
        print(f"Processing {video}")
        video_path = os.path.join(data_dir, video)
        cropped_video_path = os.path.join(cropped_dir, video)
        crop = detect_crop_params(video_path)
        crop_video_ffmpeg(video_path, cropped_video_path, crop)

        cap = cv2.VideoCapture(cropped_video_path)
        if not cap.isOpened():
            print("Error: Cannot open input video.")
            exit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in {video}: {total_frames}")

        frame_idx = 0
        stride = 5
        out_dir = os.path.join(frame_dir, video.replace(".mp4", ""))
        os.makedirs(out_dir, exist_ok=True)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                cv2.imwrite(os.path.join(out_dir, f"{frame_idx:06d}.jpg"), frame)
            frame_idx += 1
        cap.release()

# Add this to allow direct execution as a script
if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    process_videos(data_dir)
