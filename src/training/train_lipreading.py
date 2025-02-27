import os
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from jiwer import wer
import logging

from src.data.data_loader import gather_all_speakers_data, load_vocab_from_json
from src.models.transformer import LipReading3DTransformer

class TrainConfig:
    base_path = "data"

    # Split speakers: train on s1..s9, validate on s10
    train_speakers = ["s1","s2","s3","s4","s5","s6","s7","s8","s9"]
    val_speakers   = ["s10"]

    batch_size = 4
    num_workers = 4

    # Smaller Transformer
    d_model = 128
    nhead = 2
    num_encoder_layers = 2
    num_decoder_layers = 2
    dim_feedforward = 256
    dropout = 0.3
    max_len = 250

    use_ctc = False      # Disable CTC by default
    ctc_weight = 0.0     # If you want a hybrid approach, set something like 0.5

    num_epochs = 25
    learning_rate = 1e-4
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "experiments/checkpoints"
    save_prefix = "lipreading_transformer"
    print_interval = 10
    use_amp = torch.cuda.is_available()

def tokens_to_string(token_ids, vocab):
    # Convert predicted IDs to actual tokens, skipping special tokens
    special_ids = {
        vocab.sos_id, vocab.eos_id, vocab.pad_id,
        vocab.unk_id  # optionally skip <unk> if you want
    }
    words = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(words)

def validate_seq2seq(model, val_loader, vocab, cfg):
    """
    For demonstration: we do a teacher-forced validation (not true inference).
    We measure WER by comparing model's argmax at each step with the ground truth.
    """
    model.eval()
    total_wer = 0.0
    num_samples = 0

    with torch.no_grad():
        for videos, texts, lengths in val_loader:
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            with autocast(device_type=cfg.device, enabled=cfg.use_amp):
                outputs = model(videos, tgt_tokens=texts[:, :-1])

            # Teacher-forcing decode
            seq2seq_logits = outputs['seq2seq_logits']  # [B, L-1, vocab_size]
            # Argmax
            pred_ids = seq2seq_logits.argmax(dim=-1)  # [B, L-1]
            # Compare to reference (texts[:, 1:])
            for i in range(videos.shape[0]):
                pred_str = tokens_to_string(pred_ids[i].tolist(), vocab)
                gt_str = tokens_to_string(texts[i, 1:].tolist(), vocab)
                total_wer += wer(gt_str, pred_str)
                num_samples += 1

    avg_wer = total_wer / num_samples if num_samples > 0 else 1.0
    return avg_wer

def validate_ctc(model, val_loader, vocab, cfg):
    """
    Validate using the CTC head. Greedy decode.
    """
    model.eval()
    total_wer = 0.0
    num_samples = 0

    def ctc_greedy_decode(log_probs, blank_id=0):
        # log_probs: [B, T, ctc_vocab_size]
        preds = log_probs.argmax(dim=-1)  # [B, T]
        batch_tokens = []
        for i in range(preds.shape[0]):
            tokens = []
            prev = None
            for t in preds[i]:
                tid = t.item()
                if tid != blank_id and tid != prev:
                    tokens.append(tid)
                prev = tid
            batch_tokens.append(tokens)
        return batch_tokens

    with torch.no_grad():
        for videos, texts, lengths in val_loader:
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            with autocast(device_type=cfg.device, enabled=cfg.use_amp):
                outputs = model(videos, tgt_tokens=None)
                log_probs = F.log_softmax(outputs['ctc_logits'], dim=-1)

            # Greedy decode
            batch_tokens = ctc_greedy_decode(log_probs, blank_id=0)
            for i in range(len(batch_tokens)):
                pred_str = tokens_to_string(batch_tokens[i], vocab)
                gt_str   = tokens_to_string(texts[i].tolist(), vocab)
                total_wer += wer(gt_str, pred_str)
                num_samples += 1

    avg_wer = total_wer / num_samples if num_samples > 0 else 1.0
    return avg_wer

def train_lipreading_model(resume_checkpoint=None):
    cfg = TrainConfig()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    os.makedirs(cfg.save_dir, exist_ok=True)

    logging.info(f"Device: {cfg.device}, AMP: {cfg.use_amp}")

    vocab_json_path = os.path.join(cfg.base_path, "raw", "word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json_path)
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0
    logging.info(f"Loaded vocab size: {len(vocab)}")

    # Gather train data (speakers s1..s9)
    train_loader = gather_all_speakers_data(
        speaker_ids=cfg.train_speakers,
        base_path=cfg.base_path,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )
    # Gather val data (speaker s10)
    val_loader = gather_all_speakers_data(
        speaker_ids=cfg.val_speakers,
        base_path=cfg.base_path,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers
    )

    model = LipReading3DTransformer(
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

    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.learning_rate,
                           betas=(0.9, 0.98),
                           eps=1e-9,
                           weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    seq2seq_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
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
            target_lengths = lengths.to(cfg.device)

            optimizer.zero_grad()

            with autocast(device_type=cfg.device, enabled=cfg.use_amp):
                # Teacher-forcing on seq2seq branch
                outputs = model(videos, tgt_tokens=texts[:, :-1],
                                input_lengths=None,
                                target_lengths=target_lengths,
                                pad_id=pad_id)
                loss = 0.0

                # Seq2Seq loss
                if 'seq2seq_logits' in outputs:
                    decoder_target = texts[:, 1:]  # shift for teacher forcing
                    B, Lm1, V = outputs['seq2seq_logits'].shape
                    seq2seq_loss = seq2seq_criterion(
                        outputs['seq2seq_logits'].reshape(-1, V),
                        decoder_target.reshape(-1)
                    )
                    loss += (1.0 - cfg.ctc_weight) * seq2seq_loss

                # CTC loss
                if cfg.use_ctc and 'ctc_logits' in outputs:
                    log_probs = F.log_softmax(outputs['ctc_logits'], dim=-1)
                    # log_probs shape [B, T', ctc_vocab_size] => transpose for CTC [T', B, C]
                    T_prime = log_probs.shape[1]
                    log_probs = log_probs.transpose(0, 1)
                    # input_lengths = T' for each sample
                    input_lengths = torch.full((videos.size(0),), T_prime, dtype=torch.long).to(cfg.device)

                    # If any target length > T_prime, skip to avoid NaN
                    if (target_lengths > T_prime).any():
                        continue

                    ctc_loss = ctc_criterion(
                        log_probs, texts, input_lengths, target_lengths
                    )
                    loss += cfg.ctc_weight * ctc_loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN/Inf loss at epoch {epoch+1}, batch {batch_idx+1}. Skipping.")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (batch_idx + 1) % cfg.print_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                logging.info(f"Epoch [{epoch+1}/{cfg.num_epochs}], "
                             f"Step [{batch_idx+1}/{len(train_loader)}], "
                             f"Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")

        epoch_loss /= len(train_loader)
        logging.info(f"** Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} **")

        # Validation
        if cfg.use_ctc:
            val_wer = validate_ctc(model, val_loader, vocab, cfg)
        else:
            val_wer = validate_seq2seq(model, val_loader, vocab, cfg)

        logging.info(f"Validation WER: {val_wer:.4f}")
        scheduler.step(val_wer)

        # Save checkpoint
        checkpoint_path = os.path.join(cfg.save_dir, f"{cfg.save_prefix}_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
            'config': vars(cfg)
        }, checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")

    logging.info("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    train_lipreading_model(resume_checkpoint=args.resume_checkpoint)
