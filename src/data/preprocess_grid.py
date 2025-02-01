import os
import glob
import cv2
import numpy as np
import argparse
from src.utils.detect_utils import crop_video_to_mouth_array

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


def main():
    parser = argparse.ArgumentParser(description="speaker id for preprocessing")
    parser.add_argument("--spk_id", type=str, required=True, help="speaker id for preprocessing")
    args = parser.parse_args()

    # Paths
    base_path = "data"
    speaker_id = args.spk_id
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
        
        # print(f"Processed {vid_path} -> {out_npy_path} ({frames_array.shape}), transcript: '{transcript}'")
    
    print("Done preprocessing.")

if __name__ == "__main__":
    main()
