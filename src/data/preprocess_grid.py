import os
import glob
import cv2
import numpy as np
import face_recognition

"""
Script to preprocess the GRID corpus:
1) For each video in data/raw/s1/videos, find alignment in data/raw/s1/alignments.
2) Detect mouth region on the first frame (using face landmarks).
3) Crop & resize each frame to (112, 112), accumulate them in a NumPy array [T, 112, 112, 3].
4) Save the array to data/processed/s1/<video_id>_cropped.npy.
5) Save the transcript (entire utterance) in a small .txt file next to the .npy.
"""

def parse_alignment_file(align_path):
    """
    Parse the .align file to extract the full utterance.
    For typical GRID alignments, each line may look like:
       0.00 0.30 PLACE
       0.30 0.60 BLUE
       0.60 0.85 AT
       ...
    We'll gather the third column as the word (except if it is 'sil' or empty).
    Returns a single string: "place blue at ..."
    Adapt this if your alignment format differs.
    """
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # Typically: [start_time, end_time, word]
            if len(parts) == 3:
                word = parts[2]
                # GRID often has 'sil' for silence
                if word.lower() != 'sil':
                    words.append(word)
    # Join into a single utterance
    sentence = " ".join(words)
    return sentence

def get_mouth_bbox(face_landmarks):
    """
    Given the dictionary of face landmarks from face_recognition, 
    return a bounding box for the mouth region.
    """
    if 'top_lip' not in face_landmarks or 'bottom_lip' not in face_landmarks:
        return None
    
    lip_points = face_landmarks['top_lip'] + face_landmarks['bottom_lip']
    x_coords = [p[0] for p in lip_points]
    y_coords = [p[1] for p in lip_points]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    margin = 10  # Add some extra margin around the lips
    return (x_min - margin, y_min - margin, x_max + margin, y_max + margin)

def crop_video_to_mouth_array(video_path, desired_size=(112,112)):
    """
    1) Detect mouth bbox in the first frame (assuming fairly static posture in GRID).
    2) For each frame, crop mouth region, resize, and store in a list.
    3) Return a NumPy array [T, H, W, 3] in RGB.
    """
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    
    # Read first frame to detect face landmarks
    success, first_frame = cap.read()
    if not success or first_frame is None:
        print(f"Warning: Could not read first frame from {video_path}")
        cap.release()
        return None
    
    # Convert BGR->RGB for face_recognition
    rgb_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(rgb_first_frame)
    
    # If no face found, return None
    if len(face_landmarks_list) == 0:
        print(f"Warning: No face landmarks found in first frame of {video_path}")
        cap.release()
        return None
    
    # Get bounding box for the mouth from the first face detected
    mouth_bbox = get_mouth_bbox(face_landmarks_list[0])
    if not mouth_bbox:
        print(f"Warning: Could not find mouth landmarks in first frame of {video_path}")
        cap.release()
        return None
    
    x_min, y_min, x_max, y_max = mouth_bbox
    
    # Reset video to frame 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Loop over all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop mouth region
        crop = frame[y_min:y_max, x_min:x_max]
        
        # Resize to desired size (112,112)
        crop = cv2.resize(crop, desired_size)
        
        # Convert BGR->RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        frames_list.append(crop_rgb)
    
    cap.release()
    
    if len(frames_list) == 0:
        return None
    
    # Stack into NumPy array [T, H, W, 3]
    frames_array = np.stack(frames_list, axis=0)
    return frames_array

def main():
    # Paths
    base_path = "data"
    speaker_id = "s1"
    raw_dir = os.path.join(base_path, "raw", speaker_id)
    video_dir = os.path.join(raw_dir, "videos")      # e.g. data/raw/s1/videos
    align_dir = os.path.join(raw_dir, "alignments")  # e.g. data/raw/s1/alignments
    
    processed_dir = os.path.join(base_path, "processed", speaker_id)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Gather all video files in the videos folder
    video_files = glob.glob(os.path.join(video_dir, "*.mpg"))
    
    print(f"Found {len(video_files)} video files in {video_dir}")
    
    for vid_path in video_files:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        
        # Corresponding alignment file
        align_path = os.path.join(align_dir, base_name + ".align")
        
        # Parse alignment to get the transcript
        if not os.path.isfile(align_path):
            print(f"Warning: alignment file not found for {vid_path}")
            continue
        
        transcript = parse_alignment_file(align_path)
        
        # Process video -> mouth array
        frames_array = crop_video_to_mouth_array(vid_path, desired_size=(112,112))
        if frames_array is None:
            print(f"Skipping {vid_path} due to empty frames array.")
            continue
        
        # Save the .npy
        out_npy_path = os.path.join(processed_dir, f"{base_name}_cropped.npy")
        np.save(out_npy_path, frames_array)
        
        # Save the transcript
        out_txt_path = os.path.join(processed_dir, f"{base_name}_transcript.txt")
        with open(out_txt_path, "w") as f:
            f.write(transcript + "\n")
        
        print(f"Processed {vid_path} -> {out_npy_path} ({frames_array.shape}), transcript: '{transcript}'")
    
    print("Done preprocessing.")

if __name__ == "__main__":
    main()
