import cv2
import os

video_dir = "../data/datasets/s1/videos/"
frames_dir = "../data/datasets/s1/frames/"
os.makedirs(frames_dir, exist_ok=True)

def extract_all_frames(video_path, output_dir):
    """
    Extract all frames from a video and store them in the output directory.
    """
    video_name = os.path.basename(video_path).split(".")[0]  # Get the video name without extension
    video_frames_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_frames_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()
    
    while success:
        frame_filename = os.path.join(video_frames_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)  # Save each frame as an image
        success, frame = cap.read()
        frame_count += 1
    
    cap.release()

# Process all videos
for video_file in os.listdir(video_dir):
    video_path = os.path.join(video_dir, video_file)
    print(f"Processing {video_file}...")
    extract_all_frames(video_path, frames_dir)
    print(f"Frames for {video_file} stored successfully!")
