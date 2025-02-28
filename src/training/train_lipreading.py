import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from jiwer import wer
import logging

from src.data.data_loader import gather_all_speakers_data, load_vocab_from_json
from src.models.transformer import LipReadingR2Plus1DTransformer  # from your new script

class TrainConfig:
    base_path = "data"

    # Potentially speaker-dependent for highest accuracy
    train_speakers = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]
    val_speakers   = ["s1"]  # e.g. speaker-dependent or a small subset
    # NOTE: For truly speaker-independent, hold out a speaker not in the training list

    batch_size = 8          # can go bigger if you have memory
    num_workers = 4

    # Larger Transformer
    d_model = 384
    nhead = 6
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 1536
    dropout = 0.3
    max_len = 400

    use_ctc = False
    ctc_weight = 0.0

    # More epochs for big model
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "experiments/r2plus1d_ckpt"
    save_prefix = "lipreading_r2plus1d_transformer"
    print_interval = 20
    use_amp = torch.cuda.is_available()
    label_smoothing = 0.1

def tokens_to_string(token_ids, vocab):
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id, vocab.unk_id}
    words = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(words)

def validate_seq2seq(model, val_loader, vocab, cfg):
    model.eval()
    total_wer = 0.0
    num_samples = 0
    with torch.no_grad():
        for videos, texts, lengths in val_loader:
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)
            # teacher-forced forward
            with autocast(device_type=("cuda" if cfg.device.startswith("cuda") else "cpu"), enabled=cfg.use_amp):
                outputs = model(videos, tgt_tokens=texts[:, :-1], pad_id=vocab.pad_id)
            seq2seq_logits = outputs['seq2seq_logits']
            pred_ids = seq2seq_logits.argmax(dim=-1)
            for i in range(videos.shape[0]):
                gt_str = tokens_to_string(texts[i, 1:].tolist(), vocab)
                pd_str = tokens_to_string(pred_ids[i].tolist(), vocab)
                total_wer += wer(gt_str, pd_str)
                num_samples += 1
    return total_wer / num_samples if num_samples>0 else 1.0

def train_lipreading_model(resume_checkpoint=None):
    cfg = TrainConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    os.makedirs(cfg.save_dir, exist_ok=True)

    logging.info(f"Device: {cfg.device}, AMP: {cfg.use_amp}")

    vocab_json_path = os.path.join(cfg.base_path, "raw", "word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json_path)
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0
    logging.info(f"Loaded vocab size: {len(vocab)}")

    train_loader = gather_all_speakers_data(
        speaker_ids=cfg.train_speakers,
        base_path=cfg.base_path,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    val_loader = gather_all_speakers_data(
        speaker_ids=cfg.val_speakers,
        base_path=cfg.base_path,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    model = LipReadingR2Plus1DTransformer(
        vocab_size=len(vocab),
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
        use_ctc=cfg.use_ctc
    ).to(cfg.device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

    # CrossEntropy with label smoothing
    seq2seq_criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=cfg.label_smoothing)
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True) if cfg.use_ctc else None

    scaler = GradScaler(enabled=cfg.use_amp)

    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Resuming from epoch {start_epoch+1}")

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (videos, texts, lengths) in enumerate(train_loader):
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            optimizer.zero_grad()
            device_type = "cuda" if cfg.device.startswith("cuda") else "cpu"
            with autocast(device_type=device_type, enabled=cfg.use_amp):
                outputs = model(videos, tgt_tokens=texts[:, :-1], pad_id=pad_id)
                loss = 0.0

                if 'seq2seq_logits' in outputs:
                    decoder_target = texts[:, 1:]  # shift by 1
                    B, Lm1, V = outputs['seq2seq_logits'].shape
                    seq2seq_loss = seq2seq_criterion(
                        outputs['seq2seq_logits'].reshape(-1, V),
                        decoder_target.reshape(-1)
                    )
                    loss += (1.0 - cfg.ctc_weight)*seq2seq_loss

                if cfg.use_ctc and 'ctc_logits' in outputs:
                    log_probs = F.log_softmax(outputs['ctc_logits'], dim=-1)
                    Tprime = log_probs.shape[1]
                    log_probs = log_probs.transpose(0,1)
                    input_lengths = torch.full((videos.size(0),), Tprime, dtype=torch.long).to(cfg.device)
                    target_lengths = (texts != pad_id).sum(dim=1)
                    if (target_lengths > Tprime).any():
                        continue
                    ctc_loss = ctc_criterion(log_probs, texts, input_lengths, target_lengths)
                    loss += cfg.ctc_weight*ctc_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss at epoch {epoch+1}, batch {batch_idx+1}. Skipping.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (batch_idx+1) % cfg.print_interval == 0:
                avg_loss = epoch_loss / (batch_idx+1)
                elapsed = time.time() - start_time
                logging.info(f"Epoch [{epoch+1}/{cfg.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], "
                             f"Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

        epoch_loss /= len(train_loader)
        logging.info(f"** Epoch {epoch+1} finished. Avg Loss: {epoch_loss:.4f} **")

        val_wer = validate_seq2seq(model, val_loader, vocab, cfg)
        logging.info(f"Validation WER: {val_wer:.4f}")
        scheduler.step(val_wer)

        # Save
        ckpt_path = os.path.join(cfg.save_dir, f"{cfg.save_prefix}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'config': vars(cfg)
        }, ckpt_path)
        logging.info(f"Checkpoint saved at {ckpt_path}")

    logging.info("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    train_lipreading_model(resume_checkpoint=args.resume_checkpoint)
