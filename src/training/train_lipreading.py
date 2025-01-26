import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.data_loader import create_dataloader,load_vocab_from_json
from src.models.transformer import LipReading3DTransformer

###############################################################################
# 1. Training Configuration & Hyperparameters
###############################################################################

class TrainConfig:
    # Data
    base_path = "data"
    speaker_id = "s1"
    # Path to the directory with processed .npy + .txt
    processed_dir = os.path.join(base_path, "processed", speaker_id)
    batch_size = 2
    num_workers = 0  # set >0 if you want parallel data loading

    # Model / Architecture
    vocab_size = 50      # set this properly once you build your vocab
    d_model = 128
    nhead = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 256
    dropout = 0.1

    # Training
    num_epochs = 3
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing / Logging
    save_dir = "experiments/checkpoints"
    save_prefix = "lipreading_transformer"
    print_interval = 5  # how often to print training progress

###############################################################################
# 2. Training Script
###############################################################################

def train_lipreading_model():
    cfg = TrainConfig()

    # Ensure checkpoint directory exists
    os.makedirs(cfg.save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2.1 Load Vocabulary from JSON
    # -------------------------------------------------------------------------
    vocab_json_path = os.path.join(cfg.base_path,"raw",cfg.speaker_id,"word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json_path)
    
    # Update config.vocab_size to match actual size
    cfg.vocab_size = len(vocab)
    print(f"Loaded vocab of size: {cfg.vocab_size}")

    # -------------------------------------------------------------------------
    # 2.2 Create DataLoaders (train + optional val)
    # -------------------------------------------------------------------------
    train_loader = create_dataloader(
        processed_dir=cfg.processed_dir,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        add_sos_eos=True,
        num_workers=cfg.num_workers
    )

    # If you have a separate validation set, call create_dataloader on that path
    val_loader = None  # or create_dataloader("data/processed/val", vocab, ...)

    # -------------------------------------------------------------------------
    # 2.3 Initialize Model, Optimizer, Loss
    # -------------------------------------------------------------------------
    model = LipReading3DTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        max_len=250,    # adjust if you have longer sequences
        dropout=cfg.dropout
    )

    model = model.to(cfg.device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # Loss function - we typically use CrossEntropy, ignoring the pad token
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # -------------------------------------------------------------------------
    # 2.4 Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (videos, texts, lengths) in enumerate(train_loader):
            # videos: [B, C, T, H, W]
            # texts:  [B, L]
            # lengths: list of original transcript lengths (not always used here)

            videos = videos.to(cfg.device)   # float32
            texts  = texts.to(cfg.device)    # long

            optimizer.zero_grad()

            # We do an attention-based seq2seq approach:
            #  - input to the decoder typically excludes the last token
            #  - the target for computing loss excludes the first token

            # SHIFT the text by 1 for teacher forcing
            # Example:
            #   input to decoder: texts[:, :-1]
            #   target:           texts[:, 1:]
            # That means each output token is predicted from previous tokens
            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            # Forward pass
            logits = model(videos, decoder_input)  # [B, L-1, vocab_size]

            # Compute loss
            # We compare logits with decoder_target shape [B, L-1]
            # Reshape logits -> [B*(L-1), vocab_size], target -> [B*(L-1)]
            B, Lm1, V = logits.shape  # Lm1 = L-1
            loss = criterion(logits.reshape(-1, V), decoder_target.reshape(-1))

            # Backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % cfg.print_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{cfg.num_epochs}], "
                      f"Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

        # End of epoch
        epoch_loss /= len(train_loader)
        print(f"** Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} **")

        # ---------------------------------------------------------------------
        # 2.5 (Optional) Validation Loop
        # ---------------------------------------------------------------------
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, cfg)
            print(f"Validation Loss: {val_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(cfg.save_dir, f"{cfg.save_prefix}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'config': vars(cfg)
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    print("Training complete!")

###############################################################################
# 3. (Optional) Evaluate / Validation
###############################################################################

def evaluate(model, val_loader, criterion, cfg):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for (videos, texts, lengths) in val_loader:
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            logits = model(videos, decoder_input)
            B, Lm1, V = logits.shape
            loss = criterion(logits.reshape(-1, V), decoder_target.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(val_loader)

###############################################################################
# 4. Main Entry
###############################################################################

if __name__ == "__main__":
    train_lipreading_model()
