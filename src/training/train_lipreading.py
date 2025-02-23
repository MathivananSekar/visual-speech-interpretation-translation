import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from src.data.data_loader import gather_all_speakers_data, load_vocab_from_json
from src.models.transformer import LipReading3DTransformer
import logging
from jiwer import wer

class TrainConfig:
    base_path = "data"
    speaker_ids = ["s1","s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
    batch_size = 4
    num_workers = 4
    vocab_size = 50
    d_model = 128
    nhead = 2
    num_encoder_layers = 4
    num_decoder_layers = 4
    dim_feedforward = 256
    dropout = 0.1
    max_len = 250
    num_epochs = 25  # Increased for better convergence
    learning_rate = 1e-4
    weight_decay = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "experiments/checkpoints"
    save_prefix = "lipreading_transformer"
    print_interval = 10
    use_amp = torch.cuda.is_available()
    use_ctc = True
    ctc_weight = 0.5  # Joint training

def tokens_to_string(token_ids, vocab):
    special_ids = {vocab.sos_id, vocab.eos_id, vocab.pad_id, 
                   vocab.blank_id if hasattr(vocab, 'blank_id') else 0}
    words = [vocab.id_to_token(tid) for tid in token_ids if tid not in special_ids]
    return " ".join(words)

def ctc_greedy_decode(log_probs, vocab):
    predicted_ids = log_probs.argmax(dim=-1)  # [B, T']
    output_tokens = []
    for i in range(predicted_ids.shape[0]):
        tokens = []
        prev_token = None
        for t in predicted_ids[i]:
            token = t.item()
            if token != 0 and token != prev_token:
                tokens.append(token)
            prev_token = token
        output_tokens.append(tokens)
    return output_tokens

def validate(model, val_loader, vocab, cfg):
    model.eval()
    total_wer = 0.0
    num_samples = 0
    with torch.no_grad():
        for videos, texts, lengths in val_loader:
            videos = videos.to(cfg.device, dtype=torch.float32)
            texts = texts.to(cfg.device, dtype=torch.long)
            target_lengths = lengths.clone().detach().to(cfg.device)

            with autocast('cpu' if cfg.device == "cpu" else 'cuda', enabled=cfg.use_amp):
                outputs = model(videos, None)
                if 'ctc_logits' in outputs:
                    log_probs = F.log_softmax(outputs['ctc_logits'], dim=-1)  # [B, T', C]
                    ctc_tokens = ctc_greedy_decode(log_probs, vocab)
                    for i in range(videos.shape[0]):
                        pred_text = tokens_to_string(ctc_tokens[i], vocab)
                        gt_text = tokens_to_string(texts[i].tolist(), vocab)
                        total_wer += wer(gt_text, pred_text) if pred_text else 1.0
                        num_samples += 1

    return total_wer / num_samples if num_samples > 0 else float('inf')

def train_lipreading_model(resume_checkpoint=None):
    cfg = TrainConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    logging.info(f"PyTorch version: {torch.__version__}, Device: {cfg.device}, AMP: {cfg.use_amp}")
    
    vocab_json_path = os.path.join(cfg.base_path, "raw", "word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json_path)
    cfg.vocab_size = len(vocab)
    pad_id = vocab.pad_id if vocab.pad_id is not None else 0
    logging.info(f"Loaded vocab of size: {cfg.vocab_size}")
    
    train_loader = gather_all_speakers_data(
        speaker_ids=cfg.speaker_ids, base_path=cfg.base_path, vocab=vocab,
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = gather_all_speakers_data(
        speaker_ids=["s1"], base_path=cfg.base_path, vocab=vocab,
        batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    
    model = LipReading3DTransformer(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers, num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward, max_len=cfg.max_len, dropout=cfg.dropout,
        use_ctc=cfg.use_ctc
    ).to(cfg.device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.98),
                          eps=1e-9, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    seq2seq_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True) if cfg.use_ctc else None
    scaler = GradScaler('cpu', enabled=cfg.use_amp) if cfg.device == "cpu" else GradScaler('cuda', enabled=cfg.use_amp)
    
    start_epoch = 0
    if resume_checkpoint:
        checkpoint = torch.load(resume_checkpoint, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        logging.info(f"Resuming at epoch {start_epoch + 1}")
    
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (videos, texts, lengths) in enumerate(train_loader):
            videos = videos.to(cfg.device, dtype=torch.float32)
            texts = texts.to(cfg.device, dtype=torch.long)
            target_lengths = lengths.clone().detach().to(cfg.device)
            
            optimizer.zero_grad()
            
            with autocast('cpu' if cfg.device == "cpu" else 'cuda', enabled=cfg.use_amp):
                outputs = model(videos, texts[:, :-1], None, target_lengths, pad_id)
                
                loss = 0.0
                if 'seq2seq_logits' in outputs:
                    decoder_target = texts[:, 1:]
                    B, Lm1, V = outputs['seq2seq_logits'].shape
                    seq2seq_loss = seq2seq_criterion(outputs['seq2seq_logits'].reshape(-1, V), 
                                                    decoder_target.reshape(-1))
                    loss += (1 - cfg.ctc_weight) * seq2seq_loss
                
                if 'ctc_logits' in outputs:
                    log_probs = F.log_softmax(outputs['ctc_logits'], dim=-1)
                    T = log_probs.shape[1]
                    input_lengths = torch.full((videos.shape[0],), T, device=cfg.device, dtype=torch.long)
                    log_probs = log_probs.transpose(0, 1)  # [T', B, C]
                    ctc_loss = ctc_criterion(log_probs, texts, input_lengths, target_lengths)
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
                             f"Step [{batch_idx+1}/{len(train_loader)}], Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
        
        epoch_loss /= len(train_loader)
        logging.info(f"** Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f} **")
        
        # Validation
        val_wer = validate(model, val_loader, vocab, cfg)
        logging.info(f"Validation WER: {val_wer:.4f}")
        scheduler.step(val_wer)
        
        checkpoint_path = os.path.join(cfg.save_dir, f"{cfg.save_prefix}_epoch{epoch+1}.pt")
        torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'loss': epoch_loss,
                    'config': vars(cfg)}, checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")
    
    logging.info("Training complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    args = parser.parse_args()
    print(f"PyTorch version: {torch.__version__}")
    train_lipreading_model(resume_checkpoint=args.resume_checkpoint)
