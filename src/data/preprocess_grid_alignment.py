# preprocess_grid_alignment.py

import os
import glob
import numpy as np
import argparse
import json

def parse_align_file_samples(align_path):
    """
    Reads lines: start_sample end_sample label
    and returns a list of (start, end, label).
    """
    intervals = []
    with open(align_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start_s, end_s, lbl = parts
                start_s = int(start_s)
                end_s = int(end_s)
                intervals.append((start_s, end_s, lbl.lower()))
    return intervals

def build_align_array_for_audio(
    align_intervals,
    time_a,
    hop,
    label_to_id=None
):
    """
    align_intervals: list of (start_sample, end_sample, label).
    time_a: number of audio frames (the dimension of log-mel in time).
    hop: hop length in samples used in feature extraction.
    label_to_id: dict label->int
    Returns: (align_array, label_to_id).
    align_array => shape [time_a], each index is the label ID.
    """
    if label_to_id is None:
        label_to_id = {"<unk>": 0}

    half_hop = hop // 2
    align_array = np.zeros(time_a, dtype=np.int32)

    for i in range(time_a):
        center_sample = i*hop + half_hop
        label = "<unk>"
        for (st, en, lbl) in align_intervals:
            if center_sample >= st and center_sample < en:
                label = lbl
                break
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)
        align_array[i] = label_to_id[label]

    return align_array, label_to_id

def main():
    parser = argparse.ArgumentParser(description="Process alignment for each speaker based on audio frames.")
    parser.add_argument("--spk_id", type=str, required=True,
                        help="Speaker ID, e.g. s1. Must match data/raw/s1/alignments.")
    parser.add_argument("--sr", type=int, default=25000, help="Audio sample rate used in feature extraction")
    parser.add_argument("--hop", type=int, default=250, help="Hop length in samples for log-mel frames")
    parser.add_argument("--time_a", type=int, default=300, 
                        help="Number of audio frames in your log-mel, e.g. 300 for 3s at 10ms hop.")
    args = parser.parse_args()

    base_path = "data"
    speaker_id = args.spk_id
    align_dir = os.path.join(base_path, "raw", speaker_id, "alignments")
    processed_dir = os.path.join(base_path, "processed", speaker_id)
    os.makedirs(processed_dir, exist_ok=True)

    # We'll build a global label_to_id for alignment
    label_to_id = {"<unk>": 0}

    align_files = glob.glob(os.path.join(align_dir, "*.align"))
    print(f"Found {len(align_files)} alignment files in {align_dir}")
    for align_path in align_files:
        base_name = os.path.splitext(os.path.basename(align_path))[0]  # e.g. vid1
        intervals = parse_align_file_samples(align_path)
        align_array, label_to_id = build_align_array_for_audio(
            intervals,
            time_a=args.time_a,
            hop=args.hop,
            label_to_id=label_to_id
        )
        out_npy = os.path.join(processed_dir, f"{base_name}_align.npy")
        np.save(out_npy, align_array)
        print(f"Saved {out_npy}: shape={align_array.shape}")

    # Save label_to_id for future usage in model
    label_json = os.path.join(base_path, "raw", "align_label_to_id.json")
    with open(label_json, 'w') as f:
        json.dump(label_to_id, f, indent=2)
    print(f"Saved label_to_id => {label_json}")

if __name__ == "__main__":
    main()
