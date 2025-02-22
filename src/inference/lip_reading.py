import os
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from jiwer import wer
from fast_ctc_decode import beam_search

from src.models.transformer import LipReading3DTransformer
from src.data.data_loader import load_vocab_from_json
from src.utils.detect_utils import crop_video_to_mouth_array

##############################################################################
# 1. Greedy Decoding (Seq2Seq)
##############################################################################
def greedy_decode(model, frames_tensor, vocab, max_len=100, temperature=1.0):
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id
    pad_id = vocab.pad_id

    model.eval()
    device = frames_tensor.device
    batch_size = frames_tensor.size(0)

    with torch.no_grad():
        memory = model.encode_video(frames_tensor)
        decoder_input = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        output_tokens = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            decoder_output = model.decode_text(decoder_input, memory, pad_id)
            next_logits = decoder_output[:, -1, :]
            scaled_logits = next_logits / temperature
            next_token = F.log_softmax(scaled_logits, dim=-1).argmax(dim=-1)

            for i in range(batch_size):
                token = next_token[i].item()
                output_tokens[i].append(token)
                if token == eos_id:
                    break

            next_token = next_token.unsqueeze(1)
            decoder_input = torch.cat([decoder_input, next_token], dim=1)

            if all(len(tokens) > 0 and tokens[-1] == eos_id for tokens in output_tokens):
                break

        return output_tokens

##############################################################################
# 2. CTC Beam Search Decoding with fast-ctc-decode
##############################################################################
def ctc_beam_search_decode(model, frames_tensor, vocab, beam_size=10, beam_cut_threshold=0.1):
    """
    Beam search decoding for CTC using fast-ctc-decode.
    """
    model.eval()
    device = frames_tensor.device

    with torch.no_grad():
        outputs = model(frames_tensor, None)
        if 'ctc_logits' not in outputs:
            raise ValueError("Model output lacks 'ctc_logits' for CTC decoding")

        ctc_logits = outputs['ctc_logits']  # [B, T', ctc_vocab_size]
        probs = F.softmax(ctc_logits, dim=-1).cpu().numpy()  # [B, T', C], fast-ctc-decode needs probs
        batch_size = probs.shape[0]

        # Alphabet: map token IDs to strings, blank (0) first
        alphabet = [''] + [vocab.id_to_token(i) for i in range(1, len(vocab))]  # '' for blank

        output_texts = []
        for i in range(batch_size):
            seq, _ = beam_search(
                probs[i],  # [T', C] for one batch item
                alphabet,
                beam_size=beam_size,
                beam_cut_threshold=beam_cut_threshold
            )
            # Convert text back to token IDs
            tokens = []
            for char in seq.split():  # Assuming space-separated tokens
                tid = next((i for i, t in enumerate(alphabet) if t == char), None)
                if tid is not None and tid > 0:  # Skip blank (0)
                    tokens.append(tid)
            output_texts.append(tokens)

        return output_texts

##############################################################################
# 3. Convert Token IDs to String
##############################################################################
def tokens_to_string(token_ids, vocab):
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id, 
                   vocab.blank_id if hasattr(vocab, 'blank_id') else 0}
    words = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(words)

##############################################################################
# 4. Load Ground Truth Transcript
##############################################################################
def load_ground_truth(transcript_path):
    if not os.path.exists(transcript_path):
        return None
    with open(transcript_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

##############################################################################
# 5. Main Inference Function with Validation
##############################################################################
def run_inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vocab = load_vocab_from_json(args.vocab_json)
    print(f"Loaded vocab of size: {len(vocab)}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint.get('config', {})
    use_ctc = config.get('use_ctc', True)

    model = LipReading3DTransformer(
        vocab_size=len(vocab),
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 2),
        num_encoder_layers=config.get('num_encoder_layers', 4),
        num_decoder_layers=config.get('num_decoder_layers', 4),
        dim_feedforward=config.get('dim_feedforward', 256),
        max_len=config.get('max_len', 250),
        dropout=config.get('dropout', 0.1),
        use_ctc=use_ctc
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    video_dir = "data/raw/s1/videos"
    processed_dir = "data/processed/s1"
    video_files = glob.glob(os.path.join(video_dir, "*.mpg"))
    print(f"Found {len(video_files)} video files to process")

    wer_scores = []
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        transcript_path = os.path.join(processed_dir, f"{base_name}_transcript.txt")
        ground_truth = load_ground_truth(transcript_path)

        if ground_truth is None:
            print(f"Skipping {video_path}: No transcript found at {transcript_path}")
            continue

        frames_array = crop_video_to_mouth_array(video_path, target_frames=75)
        if frames_array is None:
            print(f"Error: Could not process video {video_path}")
            continue

        frames_tensor = torch.from_numpy(frames_array).float() / 255.0
        frames_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(frames_tensor, None)

            if 'seq2seq_logits' in outputs and not args.use_ctc_only:
                greedy_tokens = greedy_decode(model, frames_tensor, vocab, max_len=100, temperature=0.7)
                greedy_text = tokens_to_string(greedy_tokens[0], vocab)
                greedy_wer = wer(ground_truth, greedy_text) if greedy_text else float('inf')
                print(f"Greedy (Seq2Seq) result for {base_name}: Predicted: '{greedy_text}', "
                      f"Ground Truth: '{ground_truth}', WER: {greedy_wer:.4f}")
                wer_scores.append(greedy_wer)

            if 'ctc_logits' in outputs:
                ctc_tokens = ctc_beam_search_decode(model, frames_tensor, vocab, beam_size=10, beam_cut_threshold=0.1)
                ctc_text = tokens_to_string(ctc_tokens[0], vocab)
                ctc_wer = wer(ground_truth, ctc_text) if ctc_text else float('inf')
                print(f"CTC Beam Search result for {base_name}: Predicted: '{ctc_text}', "
                      f"Ground Truth: '{ground_truth}', WER: {ctc_wer:.4f}")
                wer_scores.append(ctc_wer)
            else:
                print(f"Warning: {base_name} lacks CTC output.")

    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
        print(f"\nProcessed {len(wer_scores)} files successfully")
        print(f"Average WER: {avg_wer:.4f}")
    else:
        print("No valid predictions generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lipreading Inference and Validation on GRID Corpus")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--vocab_json", type=str, required=True, help="Path to the vocabulary JSON file")
    parser.add_argument("--use_ctc_only", action="store_true", help="Use only CTC decoding, skip seq2seq")
    args = parser.parse_args()

    run_inference(args)