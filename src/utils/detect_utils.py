import cv2
import numpy as np
import face_recognition

def get_mouth_bbox(face_landmarks, margin=10):
    """
    Given the dictionary of face landmarks from face_recognition, 
    return a bounding box (x_min, y_min, x_max, y_max) for the mouth region.
    """
    if 'top_lip' not in face_landmarks or 'bottom_lip' not in face_landmarks:
        return None
    
    lip_points = face_landmarks['top_lip'] + face_landmarks['bottom_lip']
    x_coords = [p[0] for p in lip_points]
    y_coords = [p[1] for p in lip_points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    return (x_min - margin, y_min - margin, x_max + margin, y_max + margin)


def crop_video_to_mouth_array(video_path, desired_size=(64,64), margin=10):
    """
    1) Detect mouth bbox on the FIRST frame of the video (assuming static posture).
    2) For each frame, crop mouth region, resize, and store in a list.
    3) Return a NumPy array of shape [T, H, W, 3] in RGB.
    """
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    
    # Read first frame to detect face landmarks
    success, first_frame = cap.read()
    if not success or first_frame is None:
        print(f"[WARN] Could not read first frame from {video_path}")
        cap.release()
        return None
    
    # Convert BGR->RGB for face_recognition
    rgb_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(rgb_first_frame)
    
    if len(face_landmarks_list) == 0:
        print(f"[WARN] No face landmarks found in first frame of {video_path}")
        cap.release()
        return None
    
    # Get bounding box for the mouth from the first face detected
    mouth_bbox = get_mouth_bbox(face_landmarks_list[0], margin)
    if not mouth_bbox:
        print(f"[WARN] Could not find mouth landmarks in first frame of {video_path}")
        cap.release()
        return None
    
    x_min, y_min, x_max, y_max = mouth_bbox
    
    # Reset video to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ensure ROI is within frame boundaries
        h, w, _ = frame.shape
        x_min_clip, y_min_clip = max(0, x_min), max(0, y_min)
        x_max_clip, y_max_clip = min(w, x_max), min(h, y_max)

        crop = frame[y_min_clip:y_max_clip, x_min_clip:x_max_clip]

        # Resize to desired_size
        crop = cv2.resize(crop, desired_size)
        
        # Convert BGR->RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frames_list.append(crop_rgb)
    
    cap.release()
    
    if len(frames_list) == 0:
        return None
    
    frames_array = np.stack(frames_list, axis=0)  # [T, H, W, 3]
    return frames_array
