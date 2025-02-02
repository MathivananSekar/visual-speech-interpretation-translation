import  torch
import os
import torch.nn as nn
from src.training.metrics import compute_wer, compute_cer
from src.data.vocab import Vocab
from src.models.transformer import LipReading3DTransformer
from src.data.data_loader import create_dataloader,load_vocab_from_json
from src.training.validation import ValidateConfig


def validate_with_wer_cer(model, val_loader, cfg, vocab):
    """
    Validation loop with WER and CER computation.
    Args:
        model: Trained model for validation.
        val_loader: DataLoader for validation data.
        cfg: Validation configuration.
        vocab: Vocabulary used for decoding.
    Returns:
        dict: Validation metrics including average loss, WER, and CER.
    """
    model.eval()
    total_loss = 0.0
    total_wer = 0.0
    total_cer = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    with torch.no_grad():
        for batch_idx, (videos, texts, lengths) in enumerate(val_loader):
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            # Prepare decoder input/output
            decoder_input = texts[:, :-1]
            decoder_target = texts[:, 1:]

            # Forward pass
            logits = model(videos, decoder_input)  # [B, L-1, vocab_size]
            B, Lm1, V = logits.shape
            loss = criterion(logits.reshape(-1, V), decoder_target.reshape(-1))
            total_loss += loss.item()

            # Decode predictions
            predictions = torch.argmax(logits, dim=-1)  # [B, L-1]
            for i in range(B):
                # Convert token IDs to strings
                ref_text = " ".join([vocab.id_to_token(id) for id in texts[i].tolist() if id not in [vocab.pad_id, vocab.sos_id, vocab.eos_id]])
                hyp_text = " ".join([vocab.id_to_token(id) for id in predictions[i].tolist() if id not in [vocab.pad_id, vocab.sos_id, vocab.eos_id]])
                print(f"Ref: {ref_text} | Hyp: {hyp_text}")
                # Compute WER and CER for this sample
                total_wer += compute_wer(ref_text, hyp_text)
                total_cer += compute_cer(ref_text, hyp_text)

            # Optionally print progress
            print(f"Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.4f}")

    # Average metrics over all samples
    avg_loss = total_loss / len(val_loader)
    avg_wer = total_wer / len(val_loader.dataset)
    avg_cer = total_cer / len(val_loader.dataset)

    print(f"Validation Complete. Avg Loss: {avg_loss:.4f}, WER: {avg_wer:.2f}%, CER: {avg_cer:.2f}%")
    return {"loss": avg_loss, "wer": avg_wer, "cer": avg_cer}


if __name__ == "__main__":
   

    base_path = "data"
    speaker_id = "s1"
    processed_dir = os.path.join(base_path, "processed", speaker_id)
     # Load vocabulary
    vocab_json_path = os.path.join(base_path,"raw","word_to_idx.json")
    vocab = load_vocab_from_json(vocab_json_path)

    # Create validation DataLoader
    val_loader = create_dataloader(
        processed_dir=processed_dir,
        vocab=vocab,
        batch_size=2,
        shuffle=False,
        add_sos_eos=True,
        num_workers=0
    )

    # Load the trained model and checkpoint
    model = LipReading3DTransformer(
        vocab_size=len(vocab),
        d_model=128,
        nhead=2,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = "experiments/checkpoints/c2/lipreading_transformer_epoch6.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Validate the model
    metrics = validate_with_wer_cer(model, val_loader, ValidateConfig(), vocab)
    print(f"Final Validation Metrics: Loss = {metrics['loss']:.4f}, WER = {metrics['wer']:.2f}%, CER = {metrics['cer']:.2f}%")
