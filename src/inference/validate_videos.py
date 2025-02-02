import os
import glob
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from src.models.transformer import LipReading3DTransformer
from src.data.data_loader import load_vocab_from_json
from src.utils.detect_utils import crop_video_to_mouth_array
from src.training.metrics import compute_wer, compute_cer  # Adjust import to your project structure

def beam_decode(model, frames_tensor, vocab, beam_size=5, max_len=100):
    """
    Beam search decoding for a transformer-based lipreading model.
    
    Args:
        model: The trained LipReading3DTransformer.
        frames_tensor: Input tensor of shape [B, C, T, H, W] (assuming B=1).
        vocab: Vocabulary object with attributes: sos_id, eos_id, pad_id, and a method id_to_token.
        beam_size: The number of beams to keep at each step.
        max_len: Maximum decoding length.
        
    Returns:
        best_seq: The best decoded sequence (list of token IDs).
    """
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id
    device = frames_tensor.device

    model.eval()
    with torch.no_grad():
        # Encode the video frames to obtain visual memory.
        # memory shape: [T', B, d_model] (assume B = 1)
        memory = model.encode_video(frames_tensor)

    # Initialize beam list with one beam: ([sos_id], cumulative_log_prob)
    beams = [([sos_id], 0.0)]
    
    for step in range(max_len):
        new_beams = []
        for seq, cum_log_prob in beams:
            # If the sequence already ended with <eos>, carry it forward unchanged.
            if seq[-1] == eos_id:
                new_beams.append((seq, cum_log_prob))
                continue

            # Prepare decoder input (shape: [1, L]) from the current sequence.
            decoder_input = torch.tensor([seq], dtype=torch.long, device=device)
            
            with torch.no_grad():
                # Get decoder output: shape [B, L, vocab_size]
                decoder_output = model.decode_text(decoder_input, memory)
            # Take logits of the last token in the sequence.
            next_logits = decoder_output[:, -1, :]  # shape [1, vocab_size]
            
            # Convert logits to log-probabilities.
            log_probs = F.log_softmax(next_logits, dim=-1)  # shape [1, vocab_size]
            log_probs = log_probs.squeeze(0)  # shape [vocab_size]
            
            # Get top beam_size tokens and their log-probabilities.
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
            
            # Expand each beam candidate.
            for i in range(beam_size):
                token = topk_indices[i].item()
                new_seq = seq + [token]
                new_cum_log_prob = cum_log_prob + topk_log_probs[i].item()
                new_beams.append((new_seq, new_cum_log_prob))
        
        # Keep only the top beam_size beams (sorted by cumulative log probability in descending order).
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        
        # If all beams end with <eos>, stop early.
        if all(seq[-1] == eos_id for seq, _ in beams):
            break

    # Choose the best beam (with highest cumulative log probability).
    best_seq, best_score = beams[0]
    return best_seq

def tokens_to_string(token_ids, vocab):
    """
    Convert token IDs to text, skipping special tokens (<sos>, <eos>, <pad>).
    Adjust the join method based on whether tokens represent words (use " ".join)
    or characters (use "".join).
    """
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id}
    words = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(words)


##############################################################################
# 1. Helper: Parse alignment file to get reference text
##############################################################################
def parse_alignment_file(align_path):
    """
    Reads a .align file (like in GRID), ignoring lines with 'sil'.
    Returns the full utterance as a string (e.g., "place blue at bin")
    """
    words = []
    with open(align_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                word = parts[2]
                if word.lower() != 'sil':
                    words.append(word)
    return " ".join(words)

##############################################################################
# 2. Greedy Decode (if model has no built-in inference)
##############################################################################
def greedy_decode(model, frames_tensor, vocab, max_len=100):
    """
    1) Encode video frames -> memory
    2) Decode step-by-step with <sos> until <eos> or max_len
    """
    model.eval()
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id

    with torch.no_grad():
        memory = model.encode_video(frames_tensor)  # (B, ?, ?)
        batch_size = frames_tensor.size(0)

        # Start with <sos>
        decoder_input = torch.LongTensor([sos_id]*batch_size).unsqueeze(1).to(frames_tensor.device)
        output_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            # decode_text -> (B, L, vocab_size)
            decoder_output = model.decode_text(decoder_input,memory)
            # Take the last time-step logits
            next_logits = decoder_output[:, -1, :]  # (B, vocab_size)
            # Greedy pick
            next_token = next_logits.argmax(dim=-1)  # (B,)

            # Append
            for i in range(batch_size):
                output_tokens[i].append(next_token[i].item())

            # Stop if all ended with <eos>
            all_eos = True
            for i in range(batch_size):
                if output_tokens[i][-1] != eos_id:
                    all_eos = False
                    break

            # Add the new token to the decoder input
            next_token = next_token.unsqueeze(1)  # (B, 1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if all_eos:
                break

        return output_tokens

##############################################################################
# 3. Convert Token IDs to String
##############################################################################
def tokens_to_string(token_ids, vocab):
    """
    Skip <sos>, <eos>, <pad>. 
    If word-based: " ".join(...). 
    If char-based: "".join(...).
    """
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id}
    tokens = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(tokens)

##############################################################################
# 4. Main Validation Loop
##############################################################################
def validate_videos(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load vocab
    vocab = load_vocab_from_json(args.vocab_json)
    print(f"Loaded vocab of size {len(vocab)}")

    # Load model
    model = LipReading3DTransformer(
        vocab_size=len(vocab),
        d_model=128,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Find all video files in args.video_dir (adjust extension as needed)
    video_paths = glob.glob(os.path.join(args.video_dir, "*.mpg"))
    if not video_paths:
        print(f"No video files found in {args.video_dir}")
        return

    total_wer = 0.0
    total_cer = 0.0
    total_samples = 0

    for video_path in video_paths:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        align_path = os.path.join(args.align_dir, base_name + ".align")

        if not os.path.isfile(align_path):
            print(f"[WARN] No align file for {video_path}")
            continue

        # Parse alignment to get reference text
        ref_text = parse_alignment_file(align_path)

        # Preprocess video
        frames_array = crop_video_to_mouth_array(video_path, desired_size=(112,112))
        if frames_array is None:
            print(f"[WARN] Could not crop {video_path}, skipping.")
            continue

        # Convert [T, H, W, 3] -> [B=1, C=3, T, H, W]
        frames_array = frames_array.transpose(3, 0, 1, 2)
        frames_tensor = torch.from_numpy(frames_array).unsqueeze(0).float().to(device)

        # Inference
        with torch.no_grad():
            predicted_token_ids = beam_decode(model, frames_tensor, vocab, beam_size=5, max_len=100)

        predicted_text = tokens_to_string(predicted_token_ids, vocab)
        print(f"Inference result for {os.path.basename(video_path)}:\n{predicted_text}")

        # Compute WER / CER
        # If your compute_wer/cers return fraction, multiply by 100 if you want percentage
        sample_wer = compute_wer(ref_text, predicted_text)
        sample_cer = compute_cer(ref_text, predicted_text)

        total_wer += sample_wer
        total_cer += sample_cer
        total_samples += 1

        print(f"\nVideo: {base_name}")
        print(f" Ref: {ref_text}")
        print(f" Hyp: {predicted_text}")
        print(f" WER: {sample_wer*100:.2f}%, CER: {sample_cer*100:.2f}%")

    # Final average
    if total_samples == 0:
        print("No valid samples found for validation.")
        return

    avg_wer = (total_wer / total_samples) * 100  # if fraction to percentage
    avg_cer = (total_cer / total_samples) * 100
    print("\n==========================================")
    print(f"Validation Complete over {total_samples} videos.")
    print(f"Avg WER: {avg_wer:.2f}%, Avg CER: {avg_cer:.2f}%")

##############################################################################
# 5. Entry Point
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Lipreading Model on Multiple Videos")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory with video files")
    parser.add_argument("--align_dir", type=str, required=True, help="Directory with corresponding .align files")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--vocab_json", type=str, required=True, help="Path to vocabulary JSON file")
    args = parser.parse_args()

    validate_videos(args)
