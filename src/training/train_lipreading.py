import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.models.transformer import LipReading3DTransformer
from src.data.data_loader import gather_all_speakers_data, load_vocab_from_json

class TrainConfig:
    base_path = "data"
    speaker_ids = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
    batch_size = 8
    num_workers = 0

    d_model = 256
    nhead = 4
    num_encoder_layers = 4
    num_decoder_layers = 2
    dim_feedforward = 1024
    dropout = 0.1

    alpha_ctc = 0.5

    num_epochs = 25
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = "experiments/checkpoints"
    save_prefix = "lipreading_hybrid"
    print_interval = 10

def train_lipreading_model(resume_checkpoint=None):
    cfg = TrainConfig()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Load Vocabulary
    vocab_json_path = os.path.join(cfg.base_path, "raw", "word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json_path)
    vocab_size = len(vocab)
    print(f"Loaded vocab of size: {vocab_size} (including special tokens)")

    # DataLoader
    train_loader = gather_all_speakers_data(
        speaker_ids=cfg.speaker_ids,
        base_path=cfg.base_path,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_loader = gather_all_speakers_data(
        speaker_ids=cfg.speaker_ids,
        base_path=cfg.base_path,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    # Initialize Model
    model = LipReading3DTransformer(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout
    ).to(cfg.device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0

    # Load Checkpoint
    start_epoch = 0
    if resume_checkpoint:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training at epoch {start_epoch + 1}")

    # Training Loop
    for epoch in range(start_epoch, start_epoch + cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_ctc_loss = 0.0
        epoch_attn_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            videos, texts, frame_lengths, text_lengths = batch
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)
            B, T, _, _, _ = videos.shape

            optimizer.zero_grad()
            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            ctc_logits, attn_logits = model(videos, decoder_input)

            # CTC Loss
            ctc_log_probs = ctc_logits.permute(1, 0, 2)  # (T', B, vocab_size)
            ctc_labels = [texts[b, :text_lengths[b]] for b in range(B)]
            ctc_labels_flat = torch.cat(ctc_labels)
            ctc_input_lengths = [int(f) for f in frame_lengths]
            ctc_label_lengths = [int(l) for l in text_lengths]
            
            loss_ctc = F.ctc_loss(
                ctc_log_probs,
                ctc_labels_flat,
                ctc_input_lengths,
                ctc_label_lengths,
                blank=vocab.token_to_id("sil"),
                reduction='mean',
                zero_infinity=True
            )

            # Attention Loss
            B_attn, Lm1, V = attn_logits.shape
            attn_logits_2d = attn_logits.view(-1, V)
            attn_target_2d = decoder_target.reshape(-1)
            loss_attn = F.cross_entropy(attn_logits_2d, attn_target_2d, ignore_index=pad_id)

            # Combined Loss
            alpha = cfg.alpha_ctc
            loss = alpha * loss_ctc + (1.0 - alpha) * loss_attn

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ctc_loss += loss_ctc.item()
            epoch_attn_loss += loss_attn.item()

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

        epoch_loss /= len(train_loader)
        epoch_ctc_loss /= len(train_loader)
        epoch_attn_loss /= len(train_loader)
        print(f"** Epoch {epoch+1} finished. "
              f"Avg Loss: {epoch_loss:.4f} (CTC:{epoch_ctc_loss:.4f}, Attn:{epoch_attn_loss:.4f}) **")

        if val_loader:
            val_loss, val_ctc, val_attn = evaluate(model, val_loader, cfg, pad_id, vocab)
            print(f"[Validation] Loss={val_loss:.4f}, CTC={val_ctc:.4f}, Attn={val_attn:.4f}")

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

def evaluate(model, val_loader, cfg, pad_id, vocab):
    model.eval()
    total_loss = 0.0
    total_ctc_loss = 0.0
    total_attn_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            videos, texts, frame_lengths, text_lengths = batch
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)
            B = texts.size(0)

            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            ctc_logits, attn_logits = model(videos, decoder_input)

            # CTC Loss
            ctc_log_probs = ctc_logits.permute(1, 0, 2)
            ctc_labels = [texts[b, :text_lengths[b]] for b in range(B)]
            ctc_labels_flat = torch.cat(ctc_labels)
            ctc_input_lengths = [int(f) for f in frame_lengths]
            ctc_label_lengths = [int(l) for l in text_lengths]

            loss_ctc = F.ctc_loss(
                ctc_log_probs,
                ctc_labels_flat,
                ctc_input_lengths,
                ctc_label_lengths,
                blank=vocab.token_to_id("sil"),
                reduction='mean',
                zero_infinity=True
            )

            # Attention Loss
            B, Lm1, V = attn_logits.shape
            attn_logits_2d = attn_logits.view(-1, V)
            attn_target_2d = decoder_target.reshape(-1)
            loss_attn = F.cross_entropy(attn_logits_2d, attn_target_2d, ignore_index=pad_id)

            loss = cfg.alpha_ctc * loss_ctc + (1 - cfg.alpha_ctc) * loss_attn
            total_loss += loss.item()
            total_ctc_loss += loss_ctc.item()
            total_attn_loss += loss_attn.item()

    avg_loss = total_loss / len(val_loader)
    avg_ctc = total_ctc_loss / len(val_loader)
    avg_attn = total_attn_loss / len(val_loader)
    return avg_loss, avg_ctc, avg_attn

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    train_lipreading_model(resume_checkpoint=args.resume_checkpoint)