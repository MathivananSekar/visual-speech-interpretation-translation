import cv2
import numpy as np
import face_recognition

def get_mouth_bbox(face_landmarks, margin=10):
    """
    Given the dictionary of face landmarks from face_recognition, 
    return a bounding box (x_min, y_min, x_max, y_max) for the mouth region.
    Uses the 'top_lip' and 'bottom_lip' points.
    """
    if 'top_lip' not in face_landmarks or 'bottom_lip' not in face_landmarks:
        return None

    # Combine both lip sets into one list of points
    lip_points = face_landmarks['top_lip'] + face_landmarks['bottom_lip']
    x_coords = [p[0] for p in lip_points]
    y_coords = [p[1] for p in lip_points]

    # Compute raw bounding box around the lips
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Expand the bounding box by a margin (in pixels)
    return (x_min - margin, y_min - margin, x_max + margin, y_max + margin)

def crop_video_to_mouth_array(video_path, desired_size=(112,112), margin=10, search_frames=10):
    """
    Process a video file to extract the mouth region from each frame.
    
    Steps:
    1) Read the first frame and detect face landmarks.
    2) Compute the mouth bounding box using the landmarks and add a margin.
    3) Compute a normalization ratio based on the desired output width.
    4) For each frame, resize it using the normalization ratio and crop the mouth region.
    5) Return a NumPy array of shape [T, desired_height, desired_width, 3] in RGB.
    
    Args:
        video_path: Path to the input video.
        desired_size: Tuple (width, height) of the desired crop output.
        margin: Extra pixels added around the raw mouth bounding box.
        
    Returns:
        A NumPy array of the cropped and resized mouth frames or None if failed.
    """
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    
    found_landmarks = False
    landmarks_frame = None
    for i in range(search_frames):
        success, frame = cap.read()
        if not success or frame is None:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)
        if len(face_landmarks_list) > 0:
            landmarks_frame = frame
            found_landmarks = True
            break
    
    if not found_landmarks:
        print(f"[WARN] No face landmarks found in the first {search_frames} frames of {video_path}")
        cap.release()
        return None

    # Use the frame where landmarks were found to compute the mouth bbox
    rgb_landmarks_frame = cv2.cvtColor(landmarks_frame, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(rgb_landmarks_frame)
    mouth_bbox = get_mouth_bbox(face_landmarks_list[0], margin)
    if not mouth_bbox:
        print(f"[WARN] Could not find mouth landmarks in frame from {video_path}")
        cap.release()
        return None
    x_min, y_min, x_max, y_max = mouth_bbox

    # (Optional) Clip bounding box to frame dimensions of the landmarks frame
    h_first, w_first, _ = landmarks_frame.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w_first, x_max)
    y_max = min(h_first, y_max)
    
    # Compute normalization ratio based on the detected bbox in the selected frame
    bbox_width = x_max - x_min
    if bbox_width <= 0:
        print(f"[WARN] Invalid bounding box width in {video_path}")
        cap.release()
        return None
    desired_width, desired_height = desired_size
    normalize_ratio = desired_width / float(bbox_width)
    norm_x_min = int(x_min * normalize_ratio)
    norm_y_min = int(y_min * normalize_ratio)
    norm_x_max = int(x_max * normalize_ratio)
    norm_y_max = int(y_max * normalize_ratio)

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        new_w = int(frame.shape[1] * normalize_ratio)
        new_h = int(frame.shape[0] * normalize_ratio)
        resized_frame = cv2.resize(frame, (new_w, new_h))

        r_h, r_w, _ = resized_frame.shape
        norm_x_min_clip = max(0, norm_x_min)
        norm_y_min_clip = max(0, norm_y_min)
        norm_x_max_clip = min(r_w, norm_x_max)
        norm_y_max_clip = min(r_h, norm_y_max)
        crop = resized_frame[norm_y_min_clip:norm_y_max_clip, norm_x_min_clip:norm_x_max_clip]
        crop = cv2.resize(crop, desired_size)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frames_list.append(crop_rgb)

    cap.release()
    if len(frames_list) == 0:
        return None
    frames_array = np.stack(frames_list, axis=0)
    return frames_array

