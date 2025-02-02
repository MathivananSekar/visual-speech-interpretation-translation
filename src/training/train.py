import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import sys
from tqdm import tqdm
from src.models.backbones import LipReadingModel
from src.data.data_loader import create_dataloader, load_vocab_from_json


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

class TrainConfig:
    data_dir = "data/processed/s1"  # or a combined folder
    vocab_json = "data/raw/word_to_idx.json"
    batch_size = 2
    num_workers = 0
    lr = 1e-4
    weight_decay = 1e-5
    num_epochs = 3
    alpha_ctc = 0.5          # weighting for ctc vs attn
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "experiments/checkpoints"
    print_interval = 5
    os.makedirs(save_dir, exist_ok=True)

def train(cfg):
    # 1) Load vocab
    vocab = load_vocab_from_json(cfg.vocab_json)
    num_classes = len(vocab)  # for CTC (including blank?)
    vocab_size  = len(vocab)  # for attention decoder

    # 2) Create dataloader
    train_loader = create_dataloader(
        data_dir=cfg.data_dir,
        vocab=vocab,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers
    )

    # 3) Model
    model = LipReadingModel(
        num_classes=num_classes,
        vocab_size=vocab_size,
        hidden_dim=256,
        nhead=4,
        num_encoder_layers=4,
        num_decoder_layers=2
    ).to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_epoch = 0

    # 4) Training loop
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_loss, epoch_ctc_loss, epoch_attn_loss = 0,0,0
        t0 = time.time()
        # Wrap the train loader with tqdm to show a progress bar
        loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")

        for batch_idx, (frames, labels, frame_lengths, label_lengths) in enumerate(loader_iter):
            # frames: (B, T, 3, 64, 128)
            # labels: (B, T)
            # frame_lengths, label_lengths: list of ints
            frames = frames.to(cfg.device)
            labels = labels.to(cfg.device)

            optimizer.zero_grad()

            # For attention-based decoding, we do teacher forcing.
            # shift the labels => input (labels[:, :-1]) and target (labels[:,1:])
            dec_inp = labels[:, :-1]
            dec_tgt = labels[:, 1:]

            # forward
            ctc_logits, attn_logits = model(frames, dec_inp)

            print("CTC logits:", ctc_logits)
            print("CTC logits softmax:", torch.softmax(ctc_logits, dim=-1))

            # ctc_logits:  (B, T, num_classes)
            # attn_logits: (B, T-1, vocab_size)

            # (1) ctc_loss
            # Must permute to (T, B, C)
            ctc_log_probs = ctc_logits.permute(1, 0, 2)
            # flatten labels for ctc
            labels_flat = labels.view(-1)
            #Debug 
            print("ctc_log_probs.shape =", ctc_log_probs.shape)  # (T, B, C)
            print("labels_flat.shape =", labels_flat.shape)       # should match sum of label_lengths
            print("frame_lengths =", frame_lengths)
            print("label_lengths =", label_lengths)
            print("sum(frame_lengths) =", sum(frame_lengths))
            print("sum(label_lengths) =", sum(label_lengths))
            print("labels:", labels)

            print("Unique labels in batch:", torch.unique(labels))
            print("Blank token probability:", torch.softmax(ctc_log_probs, dim=-1)[:, :, 0].mean().item())

            loss_ctc = F.ctc_loss(
                ctc_log_probs,               # (T, B, C)
                labels_flat,                 # 1D
                frame_lengths,               # list of T
                label_lengths,               # list of T
                blank=vocab.token_to_id("sil"),                     # assume index 0 is blank
                reduction='mean',
                zero_infinity=True
            )

            # (2) attention CE
            # attn_logits => (B, T-1, vocab_size)
            # dec_tgt => (B, T-1)
            B, Lm1, V = attn_logits.shape
            attn_logits_2d = attn_logits.reshape(-1, V)
            dec_tgt_2d = dec_tgt.reshape(-1)
            loss_attn = F.cross_entropy(attn_logits_2d, dec_tgt_2d, ignore_index=0)

            # combine
            loss = cfg.alpha_ctc * loss_ctc + (1 - cfg.alpha_ctc) * loss_attn
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ctc_loss += loss_ctc.item()
            epoch_attn_loss += loss_attn.item()
            # Log every print_interval steps
            if (batch_idx + 1) % cfg.print_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                logging.info(
                    f"Epoch [{epoch+1}/{cfg.num_epochs}], "
                    f"Step [{batch_idx+1}/{len(train_loader)}], "
                    f"Avg Loss: {avg_loss:.4f}, "
                    f"CTC Loss: {loss_ctc.item():.4f}, "
                    f"Attn Loss: {loss_attn.item():.4f}"
                )
                # Also update the tqdm progress bar
                loader_iter.set_postfix(loss=f"{avg_loss:.4f}")

        # End epoch
        epoch_loss /= len(train_loader)
        logging.info(f"Epoch {epoch+1} finished in {time.time()-t0:.1f}s, Avg Loss = {epoch_loss:.4f}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch+1,
            'model_state': model.state_dict(),
            'opt_state': optimizer.state_dict(),
        }
        ckpt_path = os.path.join(cfg.save_dir, f"model_epoch{epoch+1}.pt")
        torch.save(ckpt, ckpt_path)
        logging.info(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
