import os
import argparse
import torch
import numpy as np

from torch.nn.functional import log_softmax

# Import your triple-stream model
# e.g. from src.models.model_av_align_cross import AVAlignCrossModalModel
from src.models.model_av_align_cross import AVAlignCrossModalModel

# If you have a Vocab class and a function to load it
# e.g. from src.data.data_loader import load_vocab_from_json
from src.data.data_loader import load_vocab_from_json

def tokens_to_string(token_ids, vocab):
    """
    Convert token IDs to actual words, skipping special tokens.
    """
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id, vocab.unk_id}
    words = [vocab.id_to_token(t) for t in token_ids if t not in special_ids]
    return " ".join(words)

def greedy_decode_triple_stream(
    model,
    video_tensor,   # shape [1, 3, T_v, H, W]
    audio_tensor,   # shape [1, n_mels, time_a]
    align_tensor,   # shape [1, time_l]
    vocab,
    max_len=100
):
    """
    Greedy decode for triple-stream. We do a step-by-step approach:
      - Start with <sos> token
      - For each new token, run model(...) => produce next token
      - Stop at <eos> or max_len
    """
    device = next(model.parameters()).device
    model.eval()

    sos_id = vocab.sos_id
    eos_id = vocab.eos_id
    pad_id = vocab.pad_id if vocab.pad_id else 0

    # Start tokens = [<sos>]
    decoded_tokens = [sos_id]

    with torch.no_grad():
        for step in range(max_len):
            # Convert partial sequence to a batch tensor => shape [1, step_so_far]
            partial_tokens = torch.tensor(decoded_tokens, dtype=torch.long, device=device).unsqueeze(0)
            # shape => [1, L_so_far]

            # Run forward
            # The model will produce logits => [B=1, L_so_far, vocab_size]
            logits = model(video_tensor, audio_tensor, align_tensor, partial_tokens, pad_id=pad_id)
            # We want the last step's logit => shape [1, vocab_size]
            next_logit = logits[:, -1, :]
            # Argmax
            next_token = next_logit.argmax(dim=-1).item()

            # Append to the sequence
            decoded_tokens.append(next_token)

            if next_token == eos_id:
                break

    # Convert token IDs to string
    return tokens_to_string(decoded_tokens, vocab)


def main():
    parser = argparse.ArgumentParser(description="Triple-Stream AV+Alignment Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint .pt")
    parser.add_argument("--vocab_json", type=str, required=True,
                        help="Path to the vocab JSON (word_to_idx.json)")
    parser.add_argument("--video_npy", type=str, required=True,
                        help="Path to the video .npy file (e.g. vidX_cropped.npy). Shape [T,112,112,3]")
    parser.add_argument("--audio_npy", type=str, required=True,
                        help="Path to the audio .npy file (e.g. vidX_audio.npy). Shape [n_mels,time_a]")
    parser.add_argument("--align_npy", type=str, default=None,
                        help="Path to the align .npy file if used (e.g. vidX_align.npy). shape [time_align]")
    parser.add_argument("--align_vocab_size", type=int, default=200,
                        help="Size of alignment vocabulary if using alignment. Must match or exceed training config.")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=2)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--no_align", action="store_true",
                        help="If set, skip alignment usage. Provide dummy tensor for alignment.")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("[WARN] CUDA not available, falling back to CPU.")

    # 1) Load vocab
    vocab = load_vocab_from_json(args.vocab_json)

    # 2) Load model checkpoint
    model = AVAlignCrossModalModel(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=args.max_len,
        num_decoder_layers=args.num_decoder_layers,
        align_vocab_size=args.align_vocab_size
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3) Load the .npy data
    # video => shape [T, 112, 112, 3]
    vid_array = np.load(args.video_npy)
    # shape => [T,H,W,3]. Convert to float, permute to [C,T,H,W]
    vid_tensor = torch.from_numpy(vid_array).float()
    if vid_tensor.max() > 1.0:
        vid_tensor /= 255.0
    vid_tensor = vid_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # => [1,3,T,H,W]

    aud_array = np.load(args.audio_npy)  # => [n_mels,time_a]
    aud_tensor = torch.from_numpy(aud_array).float().unsqueeze(0).to(device)  # => [1,n_mels,time_a]

    if not args.no_align and args.align_npy:
        ali_array = np.load(args.align_npy)  # => [time_align]
        align_tensor = torch.from_numpy(ali_array).long().unsqueeze(0).to(device)  # => [1,time_align]
    else:
        # If you skip alignment, create a dummy zero
        align_tensor = torch.zeros((1, 1), dtype=torch.long, device=device)

    # 4) Run inference (greedy decode)
    pred_str = greedy_decode_triple_stream(
        model, vid_tensor, aud_tensor, align_tensor, vocab,
        max_len=100  # or some max decode length
    )

    print("\n=== Inference Result ===")
    print(f"Predicted Text: {pred_str}\n")

if __name__ == "__main__":
    main()
