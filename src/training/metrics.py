import torch

###############################################################################
# 1. Levenshtein Distance (Edit Distance)
###############################################################################
def levenshtein_distance(seq1, seq2):
    """
    Compute the Levenshtein distance (edit distance) between two sequences
    seq1 and seq2 (each being a list of tokens or characters).
    Returns the minimum number of single-character edits (insertions,
    deletions or substitutions) needed to transform seq1 into seq2.
    """
    len1, len2 = len(seq1), len(seq2)
    # Create a DP table of size (len1+1) x (len2+1)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )

    return dp[len1][len2]


###############################################################################
# 2. WER & CER Computation
###############################################################################
def compute_wer(ref_str, hyp_str):
    """
    Word Error Rate: # of word-level edit operations / # of words in ref_str.
    ref_str, hyp_str: strings (each a sequence of words).
    """
    ref_words = ref_str.split()
    hyp_words = hyp_str.split()
    if len(ref_words) == 0:
        # Avoid division by zero
        return 0.0 if len(hyp_words) == 0 else 1.0

    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)

def compute_cer(ref_str, hyp_str):
    """
    Character Error Rate: # of char-level edit operations / # of chars in ref_str.
    ref_str, hyp_str: strings (each a sequence of characters).
    """
    ref_chars = list(ref_str)
    hyp_chars = list(hyp_str)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    distance = levenshtein_distance(ref_chars, hyp_chars)
    return distance / len(ref_chars)


###############################################################################
# 3. Helper: Convert Token IDs to String
###############################################################################
def tokens_to_str(token_ids, vocab):
    """
    Convert a sequence of token IDs to a string, skipping special tokens such
    as <pad>, <sos>, <eos>, <unk> if desired. Adjust as necessary based on your
    vocabulary structure.
    """
    special_tokens = {
        vocab.pad_token,
        vocab.sos_token,
        vocab.eos_token
        # You could omit <unk> from removal if you want to see it in output
        # vocab.unk_token
    }

    words = []
    for tid in token_ids:
        word = vocab.itos[tid]  # get the string for this token ID
        if word not in special_tokens:
            words.append(word)
    return " ".join(words)


###############################################################################
# 4. Putting It All Together: compute_wer_cer
###############################################################################
def compute_wer_cer(model, data_loader, vocab, cfg):
    """
    1) Runs model inference to get predicted transcripts (token IDs).
    2) Converts predictions & ground-truth token IDs to strings.
    3) Computes word error rate (WER) and character error rate (CER).
    4) Returns average WER and CER across the entire data_loader.
    """
    model.eval()
    total_wer, total_cer = 0.0, 0.0
    total_samples = 0

    with torch.no_grad():
        for videos, texts, lengths in data_loader:
            # Move data to device
            videos = videos.to(cfg.device)
            texts = texts.to(cfg.device)

            # 1) Predict transcripts from videos
            #    Depending on your model, 'model.inference' should return
            #    a list/tensor of token IDs for each sample in the batch.
            predicted_batch = model.inference(videos)

            # 2) Convert predicted & ground truth IDs to strings
            #    The shape might be (B, T) for both predicted_batch and texts
            batch_size = videos.size(0)

            for i in range(batch_size):
                pred_ids = predicted_batch[i]     # e.g. tensor of token IDs
                gt_ids   = texts[i]              # ground-truth token IDs

                pred_str = tokens_to_str(pred_ids, vocab)
                gt_str   = tokens_to_str(gt_ids, vocab)

                # 3) Calculate WER and CER
                sample_wer = compute_wer(gt_str, pred_str)
                sample_cer = compute_cer(gt_str, pred_str)

                total_wer += sample_wer
                total_cer += sample_cer
                total_samples += 1

    # 4) Average WER & CER across all samples
    #    If you prefer "total edit ops / total words/chars" approach, you
    #    can accumulate word/char counts instead. For a simple average of
    #    sample-level WER/CER, use:
    if total_samples == 0:
        return 0.0, 0.0

    avg_wer = total_wer / total_samples
    avg_cer = total_cer / total_samples

    return avg_wer, avg_cer
