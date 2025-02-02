#!/usr/bin/env python
import argparse
import cv2
import dlib
import json
import numpy as np
import torch
import torch.nn.functional as F

from src.models.backbones import LipReadingModel  # your hybrid model definition
from src.data.vocab import Vocab

# ----------------------------------------------------------------
# Set up dlib's face detector and landmark predictor
# (Make sure the path to the shape predictor is correct)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/face_landmarks/shape_predictor_68_face_landmarks.dat")

# ----------------------------------------------------------------
# Function to extract the mouth region from a frame
def extract_mouth_region(frame, face):
    landmarks = predictor(frame, face)
    # Get all 68 landmarks
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    # Extract mouth landmarks (points 48-67)
    mouth_points = np.array(points[48:68], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(mouth_points)
    mouth_roi = frame[y:y+h, x:x+w]
    # Resize to fixed dimensions (width=128, height=64)
    mouth_roi = cv2.resize(mouth_roi, (128, 64))
    return mouth_roi

# ----------------------------------------------------------------
# Process the input video and extract mouth ROIs for each frame
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale (for face detection)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            continue  # Skip frame if no face is detected
        # Use the first detected face
        face = faces[0]
        # Extract mouth ROI
        mouth_roi = extract_mouth_region(frame, face)
        # Ensure ROI is 3-channel (convert grayscale to BGR if needed)
        if len(mouth_roi.shape) == 2:
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_GRAY2BGR)
        frames.append(mouth_roi)
    cap.release()
    if len(frames) == 0:
        raise ValueError("No valid frames extracted from video.")
    # Convert to NumPy array: shape (T, H, W, C)
    frames = np.array(frames, dtype=np.uint8)
    return frames

# ----------------------------------------------------------------
# Preprocess frames for model inference
def preprocess_frames(frames):
    # Normalize to [0,1] and convert to float32
    frames = frames.astype(np.float32) / 255.0
    # Convert from (T, H, W, C) to torch.Tensor of shape (T, C, H, W)
    frames = torch.tensor(frames)
    frames = frames.permute(0, 3, 1, 2)
    # Add batch dimension -> (1, T, C, H, W)
    frames = frames.unsqueeze(0)
    return frames

# ----------------------------------------------------------------
# Greedy auto-regressive decoder for the attention branch
def greedy_decode(model, frames, max_len, sos_id, eos_id, device):
    """
    Given a preprocessed video (frames: [1, T, C, H, W]), decode a sentence.
    Uses the attention-based decoder in a greedy (autoregressive) manner.
    """
    model.eval()
    with torch.no_grad():
        # Obtain encoder outputs: shape (T, 1, hidden_dim)
        encoder_out = model.forward_encoder(frames.to(device))
        
        # Initialize decoder input with the <sos> token.
        decoder_input = torch.tensor([[sos_id]], device=device)  # shape: (1, 1)
        
        for _ in range(max_len):
            # Forward pass through the decoder.
            # decoder expects memory: (T, B, hidden_dim) and tgt tokens: (B, U)
            attn_logits = model.forward_decoder(encoder_out, decoder_input)
            # attn_logits: (1, U, vocab_size). We take the logits for the last time step.
            last_logits = attn_logits[0, -1, :]  # shape: (vocab_size,)
            # Greedy selection
            next_token = last_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)  # shape: (1, 1)
            # Append to decoder input
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            # Stop if <eos> is generated
            if next_token.item() == eos_id:
                break
        
        # Remove the initial <sos> token and return generated token IDs.
        return decoder_input[0, 1:].tolist()

# ----------------------------------------------------------------
# Main function for inference
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabulary from JSON file.
    with open(args.vocab_path, 'r') as f:
        word2idx = json.load(f)
    # Build Vocab object. (Assuming your JSON is a dict {word: index})
    # Here, we build Vocab with the keys in order of increasing index.
    # Adjust specials as needed.
    sorted_items = sorted(word2idx.items(), key=lambda x: x[1])
    tokens = [word for word, idx in sorted_items]
    specials = {'pad': '<pad>', 'unk': '<unk>', 'sos': '<sos>', 'eos': '<eos>'}
    vocab = Vocab(tokens=tokens, specials=specials)
    sos_id = vocab.sos_id
    eos_id = vocab.eos_id

    # Create the model.
    num_classes = len(vocab)   # For the CTC head.
    vocab_size = len(vocab)     # For the attention decoder.
    model = LipReadingModel(
        num_classes=num_classes,
        vocab_size=vocab_size,
        hidden_dim=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=2
    ).to(device)
    
    # Load checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Loaded checkpoint from:", args.checkpoint)
    
    # Process the input video to get frames.
    raw_frames = process_video(args.video_path)  # shape: (T, 64, 128, 3)
    frames = preprocess_frames(raw_frames)         # shape: (1, T, 3, 64, 128)
    print(f"Processed video: {raw_frames.shape[0]} frames extracted.")

    # Perform greedy decoding with a maximum decode length.
    predicted_ids = greedy_decode(model, frames, args.max_decode_len, sos_id, eos_id, device)
    
    # Convert token IDs to words.
    predicted_words = [vocab.id_to_token(idx) for idx in predicted_ids]
    sentence = " ".join(predicted_words)
    print("Predicted sentence:", sentence)

# ----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Lipreading Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint file.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to the word_idx.json file.")
    parser.add_argument("--max_decode_len", type=int, default=50,
                        help="Maximum length (in tokens) for decoding.")
    args = parser.parse_args()
    main(args)
