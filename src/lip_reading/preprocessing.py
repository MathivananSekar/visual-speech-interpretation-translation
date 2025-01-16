import cv2

def extract_frames(video_path, output_dir, frame_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, frame_size)
        cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", resized_frame)
        frame_count += 1
    cap.release()