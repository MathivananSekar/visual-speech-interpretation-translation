import os
import glob
import cv2
import numpy as np
import argparse
from src.utils.detect_utils import crop_video_to_mouth_array, parse_alignment_timestamps

def parse_alignment_file(align_path):
    words = []
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                word = parts[2]
                if word.lower() != 'sil':
                    words.append(word)
    return " ".join(words)

def main():
    parser = argparse.ArgumentParser(description="speaker id for preprocessing")
    parser.add_argument("--spk_id", type=str, required=True, help="speaker id for preprocessing")
    args = parser.parse_args()

    base_path = "data"
    speaker_id = args.spk_id
    raw_dir = os.path.join(base_path, "raw", speaker_id)
    video_dir = os.path.join(raw_dir, "videos")
    align_dir = os.path.join(raw_dir, "alignments")
    processed_dir = os.path.join(base_path, "processed", speaker_id)
    os.makedirs(processed_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(video_dir, "*.mpg"))
    print(f"Found {len(video_files)} videos in {video_dir}")

    for vid_path in video_files:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        align_path = os.path.join(align_dir, base_name + ".align")

        if not os.path.isfile(align_path):
            print(f"Warning: alignment not found for {vid_path}")
            continue

        transcript = parse_alignment_file(align_path)
        alignments = parse_alignment_timestamps(align_path, audio_sample_rate=25000)
        frames_array = crop_video_to_mouth_array(vid_path, alignments=alignments)
        if frames_array is None:
            print(f"Failed mouth crop for {vid_path}")
            continue

        out_npy_path = os.path.join(processed_dir, f"{base_name}_cropped.npy")
        np.save(out_npy_path, frames_array)

        out_txt_path = os.path.join(processed_dir, f"{base_name}_transcript.txt")
        with open(out_txt_path, "w") as f:
            f.write(transcript + "\n")

    print("Done preprocessing.")

if __name__ == "__main__":
    main()
