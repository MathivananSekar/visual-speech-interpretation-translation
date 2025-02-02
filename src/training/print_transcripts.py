import os
import numpy as np
import argparse

def collapse_labels(labels):
    """
    Collapse consecutive duplicate tokens in the label list.
    For example, if labels are:
      ['sil', 'sil', 'bin', 'bin', 'blue', 'blue', 'sil']
    this function returns:
      ['sil', 'bin', 'blue', 'sil']
    """
    if not labels:
        return []
    collapsed = [labels[0]]
    for token in labels[1:]:
        if token != collapsed[-1]:
            collapsed.append(token)
    return collapsed

def main():
    parser = argparse.ArgumentParser(
        description="Read all .npz files and print the words (transcript) for each video."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing npz files.")
    parser.add_argument("--collapse", action="store_true",
                        help="If set, collapse consecutive duplicate labels.")
    args = parser.parse_args()

    # Get all npz files in the directory
    npz_files = [f for f in os.listdir(args.data_dir) if f.endswith(".npz")]
    npz_files.sort()  # sort for consistent ordering

    if not npz_files:
        print("No npz files found in the specified directory.")
        return

    for filename in npz_files:
        filepath = os.path.join(args.data_dir, filename)
        try:
            data = np.load(filepath, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

        # Retrieve labels (expected to be an array of strings)
        labels = data["labels"]
        # Ensure labels is a list of strings:
        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        if args.collapse:
            transcript = collapse_labels(labels)
        else:
            transcript = labels

        print(f"Video: {filename}")
        print("Transcript:", " ".join(transcript))
        print("-" * 50)

if __name__ == "__main__":
    main()
