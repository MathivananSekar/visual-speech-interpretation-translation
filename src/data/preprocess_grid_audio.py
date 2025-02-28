import os
import glob
import argparse
import numpy as np
import librosa

def process_audio(
    input_path,
    output_path,
    sr=25000,
    duration=3.0,
    n_mels=40,
    hop_length=250
):
    """
    1) Load the audio from input_path using librosa.load(...).
    2) Optionally fix length to `duration` (e.g., 3.0s).
    3) Compute log-mel => shape [n_mels, time].
    4) Save as .npy in output_path.
    """

    # 1) load audio
    # If the file is shorter than `duration`, librosa.load truncates at that point. We'll pad below if needed.
    y, _ = librosa.load(input_path, sr=sr, duration=duration)

    # We want exactly sr*duration samples if possible
    needed_samples = int(sr * duration)
    if len(y) < needed_samples:
        # pad with zeros if too short
        padded = np.zeros(needed_samples, dtype=y.dtype)
        padded[:len(y)] = y
        y = padded
    else:
        # if it's longer, slice it
        y = y[:needed_samples]

    # 2) Compute mel spectrogram
    # S shape => [n_mels, time_a]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    logS = librosa.power_to_db(S, ref=np.max)

    # 3) Save
    np.save(output_path, logS)


def main():
    parser = argparse.ArgumentParser(description="Preprocess audio files for GRID corpus")
    parser.add_argument("--spk_id", type=str, required=True,
                        help="Speaker ID, e.g. s1 => data/raw/s1/audios/")
    parser.add_argument("--sr", type=int, default=25000,
                        help="Target sample rate (e.g. 25000 if your video was 25fps, 1000 samples/frame).")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Duration in seconds to keep from each clip.")
    parser.add_argument("--n_mels", type=int, default=40,
                        help="Number of mel bins.")
    parser.add_argument("--hop_length", type=int, default=250,
                        help="Samples per hop for the STFT (e.g. 250 => 10ms at sr=25000).")
    args = parser.parse_args()

    base_path = "data"
    raw_audio_dir = os.path.join(base_path, "raw", args.spk_id, "audios")
    out_dir = os.path.join(base_path, "processed", args.spk_id)
    os.makedirs(out_dir, exist_ok=True)

    audio_files = glob.glob(os.path.join(raw_audio_dir, "*.wav*"))  # check only "*.wav"
    print(f"Found {len(audio_files)} audio files in {raw_audio_dir}")

    for audio_path in audio_files:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_npy = os.path.join(out_dir, base_name + "_audio.npy")

        process_audio(
            audio_path,
            out_npy,
            sr=args.sr,
            duration=args.duration,
            n_mels=args.n_mels,
            hop_length=args.hop_length
        )
        print(f"[DONE] {audio_path} => {out_npy}")

if __name__ == "__main__":
    main()
