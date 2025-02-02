import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models.backbones import LipReadingModel

from src.data.data_loader import gather_all_speakers_data, load_vocab_from_json


###############################################################################
# 1. Training Configuration
###############################################################################

class TrainConfig:
    # Data
    base_path = "data"
    speaker_ids = ["s1"]
    batch_size = 2
    num_workers = 0  # >0 if you want multiprocessing in data loading

    # Model / Architecture
    d_model = 256
    nhead = 4
    num_encoder_layers = 4
    num_decoder_layers = 2
    dim_feedforward = 512
    dropout = 0.1

    # For the Hybrid approach:
    alpha_ctc = 0.5  # Weight for the CTC loss in the combined loss
                     # Final Loss = alpha_ctc * ctc_loss + (1-alpha_ctc) * attention_ce_loss

    # Training
    num_epochs = 3
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Checkpointing / Logging
    save_dir = "experiments/checkpoints"
    save_prefix = "lipreading_hybrid"
    print_interval = 10  # how often to print training progress


###############################################################################
# 2. Training Script
###############################################################################

def train_lipreading_model(resume_checkpoint=None):
    cfg = TrainConfig()

    # Ensure checkpoint directory exists
    os.makedirs(cfg.save_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2.1 Load Vocabulary
    # -------------------------------------------------------------------------
    vocab_json_path = os.path.join(cfg.base_path, "raw", "word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json_path)
    vocab_size = len(vocab)
    print(f"Loaded vocab of size: {vocab_size} (including special tokens)")


    # -------------------------------------------------------------------------
    # 2.2 Create DataLoader (train) + (optional) val_loader
    # -------------------------------------------------------------------------
    train_loader = gather_all_speakers_data(
        speaker_ids=cfg.speaker_ids,
        base_path=cfg.base_path,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    val_loader = None  # or define a separate gather_all_speakers_data(...) for validation

    # -------------------------------------------------------------------------
    # 2.3 Initialize Model
    # -------------------------------------------------------------------------
    model = LipReadingModel(
        num_classes=vocab_size,       # for the CTC head
        vocab_size=vocab_size,        # for the attention decoder
        hidden_dim=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_encoder_layers,  # or separate if encoder/decoder differ
        alpha=cfg.alpha_ctc           # If your model itself needs alpha; otherwise keep it here
    )
    model = model.to(cfg.device)

    # -------------------------------------------------------------------------
    # 2.4 Optimizer + Loss
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )

    # We’ll compute:
    #   (1) CTC loss via F.ctc_loss
    #   (2) Attention-based cross-entropy ignoring <pad>
    #   final_loss = alpha_ctc * ctc_loss + (1 - alpha_ctc) * cross_entropy

    pad_id = vocab.pad_id if vocab.pad_id is not None else 0

    # -------------------------------------------------------------------------
    # 2.5 Optionally Load Checkpoint
    # -------------------------------------------------------------------------
    start_epoch = 0
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training at epoch {start_epoch + 1}")

    # -------------------------------------------------------------------------
    # 2.6 Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(start_epoch, start_epoch + cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_attn_loss = 0.0

        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # We assume batch => (videos, texts, frame_lengths, text_lengths)
            # If your loader only returns (videos, texts, lengths),
            # adapt the code below to define ctc_input_lengths and ctc_label_lengths.

            videos, texts, frame_lengths, text_lengths = batch
            # videos: (B, T, 3, H, W)
            # texts:  (B, L)
            # frame_lengths: list of length B
            # text_lengths:  list of length B

            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            B, T, _, _, _ = videos.shape

            # Zero out gradients
            optimizer.zero_grad()

            # -----------------------------
            # 1) Forward pass (hybrid model)
            # -----------------------------
            # We'll do teacher forcing for attention:
            #  decoder_input = texts[:, :-1]
            #  decoder_target = texts[:, 1:]
            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            ctc_logits, attn_logits = model(videos, decoder_input)
            # ctc_logits:   (B, T, vocab_size) for CTC
            # attn_logits:  (B, L-1, vocab_size) for attention

            # -----------------------------
            # 2) Compute CTC Loss
            # -----------------------------
            # a) Permute ctc_logits to (T, B, C) for PyTorch ctc_loss
            ctc_log_probs = ctc_logits.permute(1, 0, 2)  # (T, B, vocab_size)

            # b) Flatten out the text for CTC
            #    We assume each sample's text length is text_lengths[i].
            #    input_lengths => frame_lengths
            #    label_lengths => text_lengths
            #    (If your texts contain <bos>/<eos>, you might do text_lengths[i]-1.)

            # You can pass the entire 2D texts as 1D by flattening. But typically
            # ctc_loss wants them in one 1D sequence. We'll do something minimal:
            # We'll assume texts (B, L) is the "full transcript" w/o shift.
            # If you do have <bos> or <eos>, you might want to remove them for CTC labeling.
            ctc_labels = texts  # shape (B, L)
            ctc_labels_flat = ctc_labels.contiguous().view(-1)  # (B*L,)

            # ctc_input_lengths = frame_lengths
            # ctc_label_lengths = text_lengths
            # NOTE: They must be plain Python list/tuple for ctc_loss in PyTorch
            ctc_input_lengths = [int(f) for f in frame_lengths]
            ctc_label_lengths = [int(l) for l in text_lengths]

            #Debug 
            print("ctc_log_probs.shape =", ctc_log_probs.shape)  # (T, B, C)
            print("labels_flat.shape =", ctc_labels_flat.shape)       # should match sum of label_lengths
            print("frame_lengths =", frame_lengths)
            print("label_lengths =", ctc_label_lengths)
            print("sum(frame_lengths) =", sum(frame_lengths))
            print("sum(label_lengths) =", sum(ctc_label_lengths))

            # c) Compute ctc_loss
            # By default, blank=0. Ensure that your vocab[0] is <blank>, or set blank=some_index.
            loss_ctc = F.ctc_loss(
                ctc_log_probs,          # (T, B, C)
                ctc_labels_flat,        # (sum of label lengths) => 1D
                ctc_input_lengths,      # list of T for each sample
                ctc_label_lengths,      # list of label lengths
                blank=vocab.token_to_id("sil"),                # index of blank token
                reduction='mean',
                zero_infinity=True
            )

            # -----------------------------
            # 3) Compute Attention-based CE Loss
            # -----------------------------
            # attn_logits => (B, L-1, vocab_size)
            B_attn, Lm1, V = attn_logits.shape
            attn_logits_2d = attn_logits.view(-1, V)               # (B*(L-1), vocab_size)
            attn_target_2d = decoder_target.reshape(-1)            # (B*(L-1))
            loss_attn = F.cross_entropy(attn_logits_2d, attn_target_2d, ignore_index=pad_id)

            # -----------------------------
            # 4) Combine the two losses
            # -----------------------------
            alpha = cfg.alpha_ctc
            loss = alpha * loss_ctc + (1.0 - alpha) * loss_attn

            # Backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ctc_loss += loss_ctc.item()
            epoch_attn_loss += loss_attn.item()

            # Print progress
            if (batch_idx + 1) % cfg.print_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                avg_ctc = epoch_ctc_loss / (batch_idx + 1)
                avg_attn = epoch_attn_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                current_epoch = epoch + 1
                print(f"Epoch [{current_epoch}/{start_epoch + cfg.num_epochs}], "
                      f"Step [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f} (CTC:{avg_ctc:.4f}, Attn:{avg_attn:.4f}), "
                      f"Time: {elapsed:.2f}s")

        # End of epoch
        epoch_loss /= len(train_loader)
        epoch_ctc_loss /= len(train_loader)
        epoch_attn_loss /= len(train_loader)
        print(f"** Epoch {epoch+1} finished. "
              f"Avg Loss: {epoch_loss:.4f} (CTC:{epoch_ctc_loss:.4f}, Attn:{epoch_attn_loss:.4f}) **")

        # ---------------------------------------------------------------------
        # (Optional) Validation
        # ---------------------------------------------------------------------
        if val_loader is not None:
            val_loss, val_ctc, val_attn = evaluate(model, val_loader, cfg, pad_id)
            print(f"[Validation] Loss={val_loss:.4f}, CTC={val_ctc:.4f}, Attn={val_attn:.4f}")

        # ---------------------------------------------------------------------
        # 2.7 Save checkpoint
        # ---------------------------------------------------------------------
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
# 5. (Optional) Evaluate
###############################################################################

def evaluate(model, val_loader, cfg, pad_id=0):
    model.eval()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_attn_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            videos, texts, frame_lengths, text_lengths = batch
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            # Teacher forcing
            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            ctc_logits, attn_logits = model(videos, decoder_input)
            # ctc
            ctc_log_probs = ctc_logits.permute(1, 0, 2)
            ctc_labels_flat = texts.contiguous().view(-1)
            ctc_input_lengths = [int(f) for f in frame_lengths]
            ctc_label_lengths = [int(l) for l in text_lengths]
            loss_ctc = F.ctc_loss(
                ctc_log_probs,
                ctc_labels_flat,
                ctc_input_lengths,
                ctc_label_lengths,
                blank=0,
                reduction='mean',
                zero_infinity=True
            )

            # attn
            B, Lm1, V = attn_logits.shape
            attn_logits_2d = attn_logits.view(-1, V)
            attn_target_2d = decoder_target.reshape(-1)
            loss_attn = F.cross_entropy(attn_logits_2d, attn_target_2d, ignore_index=pad_id)

            # Combine
            loss = cfg.alpha_ctc * loss_ctc + (1 - cfg.alpha_ctc) * loss_attn
            total_loss += loss.item()
            total_ctc_loss += loss_ctc.item()
            total_attn_loss += loss_attn.item()

    avg_loss = total_loss / len(val_loader)
    avg_ctc = total_ctc_loss / len(val_loader)
    avg_attn = total_attn_loss / len(val_loader)
    return avg_loss, avg_ctc, avg_attn


###############################################################################
# 6. Main Entry
###############################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to resume training from.")
    args = parser.parse_args()

    train_lipreading_model(resume_checkpoint=args.resume_checkpoint)
