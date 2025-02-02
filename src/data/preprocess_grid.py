import os
import cv2
import argparse
import numpy as np
import dlib

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/face_landmarks/shape_predictor_68_face_landmarks.dat")

# Load alignment file
def load_alignments(align_path):
    with open(align_path, 'r') as f:
        lines = f.readlines()
    alignments = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            start, end, word = parts
            # Convert the alignment times to floats
            alignments.append((float(start), float(end), word))
    # Sort by start time (just in case)
    alignments = sorted(alignments, key=lambda x: x[0])
    return alignments

# Extract mouth region using facial landmarks
def extract_mouth_region(frame, face):
    landmarks = predictor(frame, face)
    landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    # Extract mouth landmarks (points 48-67)
    mouth_points = np.array(landmarks[48:68], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(mouth_points)
    mouth_roi = frame[y:y+h, x:x+w]
    # Resize to fixed dimensions: 128x64 (width x height)
    mouth_roi = cv2.resize(mouth_roi, (128, 64))
    return mouth_roi

# Process video and extract mouth regions using scaled alignment times
def extract_mouth_region_from_video(video_path, alignments, output_dir, video_id):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video '{video_id}': total_frames = {total_frames}, fps = {fps}")

    # Determine the maximum alignment timestamp from the alignments.
    max_align = alignments[-1][1]
    print(f"Maximum alignment timestamp: {max_align}")

    frames = []
    labels = []

    # Loop over each alignment segment
    for start, end, word in alignments:
        # Scale alignment timestamps to frame indices.
        start_frame = int((start / max_align) * total_frames)
        end_frame = int((end / max_align) * total_frames)
        # print(f"\nProcessing segment for word '{word}': original start={start}, end={end} => "
        #       f"mapped to frames [{start_frame}, {end_frame})")

        # Reset video pointer to the start_frame for this segment.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"Frame read failed at frame index {frame_idx}; breaking out of this segment.")
                break

            # Convert frame to grayscale for face detection.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if len(faces) == 0:
                print(f"No face detected at frame index {frame_idx}; skipping frame.")
                continue

            # Use the first detected face.
            face = faces[0]
            mouth_roi = extract_mouth_region(frame, face)
            frames.append(mouth_roi)
            # print(f"Appending word '{word}' at frame index {frame_idx}")
            labels.append(word)

    if frames:
        frames = np.array(frames, dtype=np.uint8)
        labels = np.array(labels, dtype=object)
        save_path = os.path.join(output_dir, f"{video_id}.npz")
        np.savez(save_path, frames=frames, labels=labels)
        print(f"\nSaved processed data to {save_path}")
    else:
        print(f"\nNo frames processed for video: {video_id}")

    cap.release()

def main():
    parser = argparse.ArgumentParser(description="Speaker ID for preprocessing")
    parser.add_argument("--spk_id", type=str, required=True, help="Speaker ID for preprocessing")
    args = parser.parse_args()

    base_path = "data"
    speaker_id = args.spk_id
    raw_dir = os.path.join(base_path, "raw", speaker_id)
    video_dir = os.path.join(raw_dir, "videos")
    align_dir = os.path.join(raw_dir, "alignments")
    processed_dir = os.path.join(base_path, "processed", speaker_id)
    os.makedirs(processed_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if video_file.endswith(".mpg"):
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_dir, video_file)
            align_path = os.path.join(align_dir, f"{video_id}.align")

            if os.path.exists(align_path):
                alignments = load_alignments(align_path)
                print(f"\nAlignments for video {video_id}: {alignments}")
                extract_mouth_region_from_video(video_path, alignments, processed_dir, video_id)
            else:
                print(f"Alignment file does not exist for video: {video_id}")

if __name__ == "__main__":
    main()
