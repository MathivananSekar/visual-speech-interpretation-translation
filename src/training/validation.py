import os
import torch
import torch.nn as nn
from src.data.data_loader import create_dataloader
from src.data.vocab import Vocab
from src.models.transformer import LipReading3DTransformer

###############################################################################
# 1. Validation Configuration
###############################################################################

class ValidateConfig:
    # Data
    base_path = "data"
    speaker_id = "s1"
    processed_dir = os.path.join(base_path, "processed", speaker_id)  # Path to validation data
    batch_size = 2
    num_workers = 0

    # Model / Architecture (same as training configuration)
    vocab_size = 50
    d_model = 128
    nhead = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 256
    dropout = 0.1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpoints
    checkpoint_path = "experiments/checkpoints/lipreading_transformer_epoch3.pt"

###############################################################################
# 2. Validation Function
###############################################################################

def validate_model():
    cfg = ValidateConfig()

    # Ensure checkpoint exists
    if not os.path.isfile(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {cfg.checkpoint_path}")

    # -------------------------------------------------------------------------
    # 2.1 Load Vocabulary
    # -------------------------------------------------------------------------
    tokens = ["place", "blue", "at", "red", "green", "two", "one", "please"]
    specials = {
        "pad": "<pad>",
        "unk": "<unk>",
        "sos": "<sos>",
        "eos": "<eos>"
    }
    vocab = Vocab(tokens=tokens, specials=specials)
    cfg.vocab_size = len(vocab)

    # -------------------------------------------------------------------------
    # 2.2 Create Validation DataLoader
    # -------------------------------------------------------------------------
    val_loader = create_dataloader(
        processed_dir=cfg.processed_dir,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=False,  # No shuffling for validation
        add_sos_eos=True,
        num_workers=cfg.num_workers
    )

    # -------------------------------------------------------------------------
    # 2.3 Load Model and Checkpoint
    # -------------------------------------------------------------------------
    model = LipReading3DTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        max_len=250,
        dropout=cfg.dropout
    ).to(cfg.device)

    # Load the checkpoint
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {cfg.checkpoint_path}")

    # Loss function
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # -------------------------------------------------------------------------
    # 2.4 Validation Loop
    # -------------------------------------------------------------------------
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, (videos, texts, lengths) in enumerate(val_loader):
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            # Prepare input/output for validation
            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            # Forward pass
            logits = model(videos, decoder_input)  # [B, L-1, vocab_size]
            B, Lm1, V = logits.shape
            loss = criterion(logits.reshape(-1, V), decoder_target.reshape(-1))
            total_loss += loss.item()

            # Optionally print progress
            print(f"Validation Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.4f}")

    # Average validation loss
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Complete. Average Loss: {avg_loss:.4f}")

    return avg_loss

###############################################################################
# 3. Main Entry
###############################################################################

if __name__ == "__main__":
    avg_val_loss = validate_model()
    print(f"Final Validation Loss: {avg_val_loss:.4f}")
