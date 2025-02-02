import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F

from src.models.transformer import LipReading3DTransformer
from src.data.data_loader import load_vocab_from_json
from src.utils.detect_utils import crop_video_to_mouth_array


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
# 1. Greedy Decoding Function
##############################################################################
def greedy_decode(model, frames_tensor, vocab, max_len=100):
    """
    A simple auto-regressive greedy decode for a transformer-based model.
    
    Args:
        model: The trained LipReading3DTransformer.
        frames_tensor: Input tensor of shape [B, C, T, H, W].
        vocab: Vocabulary object with attributes (sos_id, eos_id, pad_id, id_to_token).
        max_len: Maximum decoding steps.
        
    Returns:
        A list (per sample) of predicted token IDs.
    """
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id

    model.eval()
    with torch.no_grad():
        # Encode the video frames to obtain visual memory.
        memory = model.encode_video(frames_tensor)
        batch_size = frames_tensor.size(0)
        # Initialize the decoder with the <sos> token.
        decoder_input = torch.LongTensor([sos_id] * batch_size).unsqueeze(1).to(frames_tensor.device)
        print("decoder_input dtype:", decoder_input.dtype)
        output_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            # The decode_text function expects (tgt_tokens, memory)
            decoder_output = model.decode_text(decoder_input, memory)  # [B, L, vocab_size]
            next_logits = decoder_output[:, -1, :]  # Take logits for the last time step, shape [B, vocab_size]
            next_token = next_logits.argmax(dim=-1)  # Greedy selection, shape [B]

            # Append the selected token to each sample's output.
            for i in range(batch_size):
                output_tokens[i].append(next_token[i].item())

            # Check if all samples have generated the EOS token.
            all_eos = all(output_tokens[i][-1] == eos_id for i in range(batch_size))
            # Update the decoder input by appending the new token.
            next_token = next_token.unsqueeze(1)  # [B, 1]
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if all_eos:
                break

        return output_tokens

##############################################################################
# 2. Convert Token IDs to Readable String
##############################################################################
def tokens_to_string(token_ids, vocab):
    """
    Convert token IDs to text, skipping <sos>, <eos>, <pad> tokens.
    Change the join method depending on whether tokens represent words or characters.
    """
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id}
    words = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(words)

##############################################################################
# 3. Main Inference Function
##############################################################################
def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load Vocabulary.
    vocab = load_vocab_from_json(args.vocab_json)
    print(f"Loaded vocab of size: {len(vocab)}")

    # 2) Load the trained model.
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

    # 3) Preprocess Video.
    frames_array = crop_video_to_mouth_array(args.video_path, desired_size=(112,112))
    if frames_array is None:
        print(f"Error: could not process video {args.video_path}")
        return

    # Convert from [T, H, W, 3] to [B=1, C=3, T, H, W]
    frames_array = frames_array.transpose(3, 0, 1, 2)
    frames_tensor = torch.from_numpy(frames_array).unsqueeze(0).float().to(device)

    # 4) Run Model Inference using greedy decoding.
    with torch.no_grad():
        predicted_token_ids = beam_decode(model, frames_tensor, vocab, beam_size=5, max_len=100)

    predicted_text = tokens_to_string(predicted_token_ids, vocab)
    print(f"Inference result for {os.path.basename(args.video_path)}:\n{predicted_text}")

##############################################################################
# 4. Entry Point
##############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lipreading Inference on a Single Video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--vocab_json", type=str, required=True, help="Path to the vocabulary JSON file")
    args = parser.parse_args()

    run_inference(args)
