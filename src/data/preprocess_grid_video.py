import os
import glob
import cv2
import numpy as np
import argparse

from src.utils.detect_utils import crop_video_to_mouth_array, parse_alignment_timestamps


def main():
    parser = argparse.ArgumentParser(description="Preprocess videos using mouth detection/cropping.")
    parser.add_argument("--spk_id", type=str, required=True, help="Speaker ID, e.g., s1.")
    parser.add_argument("--target_frames", type=int, default=75, help="Number of frames to resample to.")
    parser.add_argument("--alignments", action='store_true',
                        help="If set, we look for alignment info. (You must define how you parse it.)")
    args = parser.parse_args()

    base_path = "data"
    raw_dir = os.path.join(base_path, "raw", args.spk_id)
    video_dir = os.path.join(raw_dir, "videos")
    align_dir = os.path.join(raw_dir, "alignments")
    processed_dir = os.path.join(base_path, "processed", args.spk_id)
    os.makedirs(processed_dir, exist_ok=True)

    vid_files = glob.glob(os.path.join(video_dir, "*.mpg"))
    print(f"Found {len(vid_files)} .mpg files in {video_dir}")

    for vid_path in vid_files:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        align_path = os.path.join(align_dir, base_name + ".align")

        if not os.path.isfile(align_path):
            print(f"Warning: alignment not found for {vid_path}")
            continue
        alignments = parse_alignment_timestamps(align_path, audio_sample_rate=25000)
        out_npy = os.path.join(processed_dir, base_name + "_cropped.npy")
        frames_array = crop_video_to_mouth_array(
            vid_path,
            alignments=alignments,
            target_frames=args.target_frames,
            debug_save=False
        )
        if frames_array is None:
            print(f"[ERROR] Skipping {vid_path} due to error.")
            continue

        np.save(out_npy, frames_array)
        print(f"[SAVED] {out_npy} => shape {frames_array.shape}")

if __name__ == "__main__":
    main()
