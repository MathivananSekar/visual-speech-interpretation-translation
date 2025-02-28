# train_av_align_cross.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from jiwer import wer
import logging

from src.data.data_loader import load_vocab_from_json
from src.data.data_loader_av_align import gather_all_speakers_data
from src.models.model_av_align_cross import AVAlignCrossModalModel

class TrainConfig:
    base_path = "data"
    train_speakers = ["s1"]  # adjust as needed
    val_speakers   = ["s1"]
    batch_size = 4
    num_workers = 2
    d_model = 256
    nhead = 4
    num_encoder_layers = 2
    num_decoder_layers = 4
    dim_feedforward = 1024
    dropout = 0.1
    max_len = 1000
    align_vocab_size = 200  # Must be >= the number of distinct alignment labels
    num_epochs = 20
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "experiments/align_cross"
    save_prefix = "av_align_cross"
    print_interval = 20
    use_amp = torch.cuda.is_available()

def tokens_to_string(token_ids, vocab):
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id, vocab.unk_id}
    words = [vocab.id_to_token(t) for t in token_ids if t not in special_ids]
    return " ".join(words)

def validate(model, loader, vocab, cfg):
    model.eval()
    total_wer = 0.0
    total_samples = 0
    with torch.no_grad():
        for vids, auds, aligns, text_batch, lengths in loader:
            videos = torch.stack(vids, dim=0).to(cfg.device)
            audios = torch.stack(auds, dim=0).to(cfg.device)
            aligns = torch.stack(aligns, dim=0).to(cfg.device)
            texts  = text_batch.to(cfg.device)

            with autocast(device_type=cfg.device, enabled=cfg.use_amp):
                logits = model(videos, audios, aligns, texts[:, :-1], pad_id=vocab.pad_id)
            pred = logits.argmax(dim=-1)  # => [B, L-1]

            B = videos.size(0)
            for i in range(B):
                ref_str = tokens_to_string(texts[i,1:].tolist(), vocab)
                hyp_str = tokens_to_string(pred[i].tolist(), vocab)
                total_wer += wer(ref_str, hyp_str)
                total_samples += 1

    return total_wer / total_samples if total_samples>0 else 1.0

def train_av_align_cross(resume_checkpoint=None):
    cfg = TrainConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    os.makedirs(cfg.save_dir, exist_ok=True)

    # Load transcript vocab
    vocab_json = os.path.join(cfg.base_path, "raw", "word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json)
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0

    # Build train/val loaders
    train_loader = gather_all_speakers_data(
        cfg.train_speakers,
        cfg.base_path,
        vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_loader = gather_all_speakers_data(
        cfg.val_speakers,
        cfg.base_path,
        vocab,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    # Build model
    model = AVAlignCrossModalModel(
        vocab_size=len(vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        num_decoder_layers=cfg.num_decoder_layers,
        align_vocab_size=cfg.align_vocab_size
    ).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    if resume_checkpoint:
        ckp = torch.load(resume_checkpoint, map_location=cfg.device)
        model.load_state_dict(ckp['model_state_dict'])
        optimizer.load_state_dict(ckp['optimizer_state_dict'])
        start_epoch = ckp['epoch']
        logging.info(f"Resumed from epoch {start_epoch+1}")

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        t0 = time.time()
        for batch_idx, (vids, auds, aligns, text_batch, lengths) in enumerate(train_loader):
            videos = torch.stack(vids, dim=0).to(cfg.device)
            audios = torch.stack(auds, dim=0).to(cfg.device)
            ali_ids = torch.stack(aligns, dim=0).to(cfg.device)
            texts  = text_batch.to(cfg.device)

            optimizer.zero_grad()
            with autocast(device_type=cfg.device, enabled=cfg.use_amp):
                logits = model(videos, audios, ali_ids, texts[:, :-1], pad_id=pad_id)
                # => [B, L-1, vocab_size]
                target = texts[:, 1:]  # shift
                B_, Lm1, V = logits.shape
                loss = criterion(logits.reshape(-1, V), target.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("Skipping nan/inf loss at epoch {}, batch {}".format(epoch+1, batch_idx+1))
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            steps += 1

            if (batch_idx+1) % cfg.print_interval == 0:
                avg_loss = epoch_loss / steps
                elapsed = time.time() - t0
                logging.info(f"Epoch [{epoch+1}/{cfg.num_epochs}], "
                             f"Step [{batch_idx+1}/{len(train_loader)}], "
                             f"Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

        if steps > 0:
            epoch_loss /= steps
        logging.info(f"** Epoch {epoch+1} finished. Avg Loss: {epoch_loss:.4f} **")

        val_wer = validate(model, val_loader, vocab, cfg)
        logging.info(f"Validation WER: {val_wer:.4f}")

        ckpt_path = os.path.join(cfg.save_dir, f"{cfg.save_prefix}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, ckpt_path)
        logging.info(f"Checkpoint saved => {ckpt_path}")

    logging.info("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    train_av_align_cross(resume_checkpoint=args.resume_checkpoint)
