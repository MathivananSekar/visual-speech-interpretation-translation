import os
import argparse
import torch
import numpy as np

from src.models.transformer import LipReading3DTransformer
from src.data.data_loader import load_vocab_from_json
from src.utils.detect_utils import crop_video_to_mouth_array

##############################################################################
# 1. Greedy Decoding (If Your Model Doesn't Have Built-In Inference)
##############################################################################
def greedy_decode(model, frames_tensor, vocab, max_len=100):
    """
    A simple auto-regressive greedy decode for a transformer-based model.
    - frames_tensor: (B, C, T, H, W)
    Returns:
        A list of token IDs (for B=1, we just return one list).
    """
    # Typically, you need an <sos> token to start, <eos> token to end.
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id

    model.eval()
    with torch.no_grad():
        # Encode the video frames (depends on your model's structure)
        # We'll assume you have something like model.encode(...) if you
        # separated encoder & decoder. If not, adapt to your forward method.
        memory = model.encode_video(frames_tensor)

        # Initialize decoder input with <sos>
        batch_size = frames_tensor.size(0)
        decoder_input = torch.LongTensor([sos_id] * batch_size).unsqueeze(1).to(frames_tensor.device)
        print("decoder_input dtype:", decoder_input.dtype)
        output_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            # decode shape might be (B, L, vocab_size)
            decoder_output = model.decode_text(decoder_input,memory)  # (B, L, vocab_size)
            next_logits = decoder_output[:, -1, :]                # last time step: (B, vocab_size)
            next_token = next_logits.argmax(dim=-1)              # (B,)

            # Append tokens
            for i in range(batch_size):
                output_tokens[i].append(next_token[i].item())

            # If all ended with EOS, can break early
            all_eos = True
            for i in range(batch_size):
                if output_tokens[i][-1] != eos_id:
                    all_eos = False
                    break

            # update decoder_input
            next_token = next_token.unsqueeze(1)  # shape (B, 1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if all_eos:
                break

        return output_tokens

##############################################################################
# 2. Convert Token IDs to a Readable String
##############################################################################
def tokens_to_string(token_ids, vocab):
    """
    Convert token IDs to text, skipping <sos>, <eos>, <pad> tokens.
    If your tokens are word-level, " ".join is typical.
    If your tokens are character-level, "".join might be better.
    """
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id}
    # Example: word-level
    words = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(words)

##############################################################################
# 3. Main Inference Function
##############################################################################
def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load Vocab
    vocab = load_vocab_from_json(args.vocab_json)
    print(f"Loaded vocab of size: {len(vocab)}")

    # 2) Load Model
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

    # 3) Preprocess Video -> (T, H, W, 3)
    frames_array = crop_video_to_mouth_array(args.video_path, desired_size=(64,64))
    if frames_array is None:
        print(f"Error: could not process video {args.video_path}")
        return

    # Convert shape [T, H, W, 3] -> torch tensor [B=1, C=3, T, H, W]
    frames_array = frames_array.transpose(3, 0, 1, 2)  # (3, T, H, W)
    frames_tensor = torch.from_numpy(frames_array).unsqueeze(0).float().to(device)

    # 4) Run Model Inference
    # If your model has a built-in `model.inference()`, use that. Otherwise:
    # We do a 2-stage approach: model.encode(...) + model.decode(...) in greedy_decode.
    with torch.no_grad():
        # If you have "model.inference": 
        # predicted_batch = model.inference(frames_tensor)
        # else do:
        predicted_batch = greedy_decode(model, frames_tensor, vocab, max_len=100)

    predicted_tokens = predicted_batch[0]  # for B=1
    predicted_text = tokens_to_string(predicted_tokens, vocab)

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
