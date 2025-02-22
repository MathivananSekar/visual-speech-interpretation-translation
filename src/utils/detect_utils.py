import os
import cv2
import numpy as np
import dlib
from scipy.interpolate import interp1d

# Initialize Dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/face_landmarks/shape_predictor_68_face_landmarks.dat")  # Ensure this file exists

def extract_mouth_region(frame, face):
    """
    Extracts the mouth region using Dlib's facial landmarks and returns a resized crop.
    """
    landmarks = predictor(frame, face)
    landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    
    # Extract mouth landmarks (points 48-67)
    mouth_points = np.array(landmarks[48:68], dtype=np.int32)
    
    # Compute bounding box around mouth landmarks
    x, y, w, h = cv2.boundingRect(mouth_points)
    
    # Extract ROI (Region of Interest)
    mouth_roi = frame[y:y+h, x:x+w]
    
    # Resize to fixed dimensions: 128x64 (width x height)
    mouth_roi = cv2.resize(mouth_roi, (128, 64))
    
    return mouth_roi

def parse_alignment_timestamps(align_path, audio_sample_rate=25000):
    """
    Parse .align file where values are audio sample indices at 25 kHz, converting to seconds.
    """
    alignments = {}
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2].lower() != 'sil':
                try:
                    start_sample, end_sample, word = float(parts[0]), float(parts[1]), parts[2]
                    start_time = start_sample / audio_sample_rate  # Convert to seconds
                    end_time = end_sample / audio_sample_rate
                    if start_time < end_time and start_time >= 0:
                        alignments[word] = (start_time, end_time)
                    else:
                        print(f"[WARN] Invalid sample range in {align_path}: {line.strip()}")
                except ValueError:
                    print(f"[WARN] Malformed line in {align_path}: {line.strip()}")
    if not alignments:
        print(f"[WARN] No valid alignments found in {align_path}")
    else:
        print(f"[INFO] Parsed {align_path}: {len(alignments)} tokens, "
              f"samples {min(t[0]*audio_sample_rate for t in alignments.values())}-"
              f"{max(t[1]*audio_sample_rate for t in alignments.values())}")
    return alignments

def crop_video_to_mouth_array(video_path, alignments=None, target_frames=75, debug_save=False):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if abs(fps - 25) > 1:
        print(f"[WARN] Video FPS ({fps}) differs from expected 25. Using 25 FPS.")
        fps = 25

    video_duration = total_frames / fps
    print(f"[DEBUG] {video_path}: total_frames={total_frames}, fps={fps}, duration={video_duration}s")

    if total_frames == 0:
        print(f"[ERROR] No frames in {video_path}. Skipping.")
        cap.release()
        return None

    frames_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            if frames_list:
                frames_list.append(frames_list[-1])
            frame_count += 1
            continue
        face = faces[0]
        mouth_roi = extract_mouth_region(frame, face)
        mouth_roi = cv2.resize(mouth_roi, (112, 112))
        frames_list.append(mouth_roi)
        frame_count += 1

    cap.release()

    if len(frames_list) == 0:
        print(f"[ERROR] No valid frames processed for {video_path}.")
        return None

    frames_array = np.stack(frames_list, axis=0)

    if alignments:
        start_time = min(t[0] for t in alignments.values())  # In seconds
        end_time = max(t[1] for t in alignments.values())
        print(f"[DEBUG] Alignments: start_time={start_time}s, end_time={end_time}s")

        if end_time > video_duration + 0.1:  # Allow small tolerance
            print(f"[WARN] Alignment ({start_time}-{end_time}s) exceeds video ({video_duration}s). Using full video.")
            start_frame, end_frame = 0, total_frames
        else:
            start_frame = max(0, int(start_time * fps))  # Convert seconds to frames
            end_frame = min(total_frames, int(end_time * fps))
            if end_frame <= start_frame:
                print(f"[WARN] Invalid range ({start_frame}-{end_frame}). Using full video.")
                start_frame, end_frame = 0, total_frames
            else:
                frames_array = frames_array[start_frame:end_frame]
                print(f"[INFO] Cropped to alignment: {start_frame}-{end_frame} frames")

        orig_frames = frames_array.shape[0]
    else:
        orig_frames = frames_array.shape[0]

    if orig_frames != target_frames:
        x_old = np.linspace(0, orig_frames - 1, orig_frames)
        x_new = np.linspace(0, orig_frames - 1, target_frames)
        interpolator = interp1d(x_old, frames_array, axis=0, kind='linear', fill_value="extrapolate")
        frames_array = interpolator(x_new)
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)

    print(f"[INFO] Processed {video_path}: {orig_frames} -> {target_frames} frames")
    return frames_array


