from jiwer import wer, cer

def compute_wer(reference, hypothesis):
    """
    Compute Word Error Rate (WER) between the reference and hypothesis.
    Args:
        reference (str): Ground truth text.
        hypothesis (str): Predicted text by the model.
    Returns:
        float: WER as a percentage.
    """
    return wer(reference, hypothesis)

def compute_cer(reference, hypothesis):
    """
    Compute Character Error Rate (CER) between the reference and hypothesis.
    Args:
        reference (str): Ground truth text.
        hypothesis (str): Predicted text by the model.
    Returns:
        float: CER as a percentage.
    """
    return cer(reference, hypothesis)
