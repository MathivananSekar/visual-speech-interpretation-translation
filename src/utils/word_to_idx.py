import os
import json

def build_word_to_index(align_dirs, save_path="word_to_idx.json"):
    """
    Scans all alignment files in multiple directories to collect unique tokens.
    Creates a dictionary mapping each token to an integer index.
    Saves the resulting dictionary to 'save_path' in JSON format.

    Args:
        align_dirs (list of str): List of directories containing the .align files.
        save_path (str): Path to save the resulting dictionary in JSON format.
    """
    tokens = set()

    # 1. Collect all tokens from alignment files in all directories
    for align_dir in align_dirs:
        for filename in os.listdir(align_dir):
            if filename.endswith(".align"):
                file_path = os.path.join(align_dir, filename)
                with open(file_path, "r") as f:
                    for line in f:
                        _, _, token = line.strip().split()
                        print(f"tokens found {token}")
                        tokens.add(token)

    # 2. Sort tokens for consistency
    sorted_tokens = sorted(tokens)

    # 3. Create a word-to-index dictionary
    #    We'll also reserve an index (e.g., 0) for <UNK> or <PAD> if needed
    word_to_idx = {}
    idx = 0
    for token in sorted_tokens:
        # Skip or handle special tokens if needed
        # e.g., if token == "sil": continue  # but usually you'd keep 'sil'
        word_to_idx[token] = idx
        idx += 1

    # 4. Save the dictionary to a JSON file
    with open(save_path, "w") as json_file:
        json.dump(word_to_idx, json_file, indent=4)

    print(f"Word-to-Index dictionary created with {len(word_to_idx)} entries.")
    print(f"Saved to: {save_path}")


# Example usage:
if __name__ == "__main__":
    # List of directories containing .align files
    alignments_dirs = [
        "data/raw/s1/alignments",
        "data/raw/s2/alignments",
        "data/raw/s3/alignments",
        "data/raw/s4/alignments",
        "data/raw/s5/alignments",
        "data/raw/s6/alignments",
        "data/raw/s7/alignments",
        "data/raw/s8/alignments",
        "data/raw/s9/alignments",
        "data/raw/s10/alignments"
    ]
    output_dict_path = "data/raw/word_to_idx.json"

    build_word_to_index(alignments_dirs, output_dict_path)