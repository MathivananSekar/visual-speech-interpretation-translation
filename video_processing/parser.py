import os
import shutil
import concurrent.futures
import logging

video_dir = "../data/datasets/s1/videos/"
audio_dir = "../data/datasets/s1/audios/"
align_dir = "../data/datasets/s1/alignments"
frame_dir = "../data/datasets/s1/frames"
output_dir = "../data/datasets/s1/labeled_data"

def process_alignment(align_file, audio_path, frame_dir, output_dir):
    """
    Process alignment file to create labeled data.
    """
    try:
        print(f"Processing align file  {align_file}")
        with open(align_file, "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            start_time, end_time, text = line.strip().split()
            start_time, end_time = float(start_time), float(end_time)
            
            # Extract corresponding audio segment
            extract_audio_segment(audio_path, start_time, end_time, f"{output_dir}/segment_{i}.wav")
            
            # Extract corresponding frames
            extract_frames_segment(frame_dir, start_time, end_time, f"{output_dir}/frames_segment_{i}")
            
            # Save text label
            with open(f"{output_dir}/segment_{i}_label.txt", "w") as label_file:
                label_file.write(text)
    except Exception as e:
        logging.error(f"Error processing alignment file {align_file}: {e}")

def extract_audio_segment(audio_path, start_time, end_time, output_path):
    """
    Extract a segment of audio between start_time and end_time.
    """
    from pydub import AudioSegment
    audio = AudioSegment.from_wav(audio_path)
    segment = audio[start_time * 1000:end_time * 1000]
    segment.export(output_path, format="wav")

def extract_frames_segment(frame_dir, start_time, end_time, output_path):
    """
    Extract frames corresponding to a specific time segment.
    """
    start_frame = int(start_time * 30)  # Assuming 30 FPS
    end_frame = int(end_time * 30)
    
    os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
    
    frame_paths = [os.path.normpath(os.path.join(frame_dir, f"frame_{frame_number:04d}.jpg")) for frame_number in range(start_frame, end_frame + 1) if os.path.exists(os.path.normpath(os.path.join(frame_dir, f"frame_{frame_number:04d}.jpg")))]
    
    for frame_path in frame_paths:
        target_path = os.path.normpath(os.path.join(output_path, os.path.basename(frame_path)))
        shutil.copy2(frame_path, target_path)

def main():
    align_files = [os.path.join(align_dir, f) for f in os.listdir(align_dir) if f.endswith(".align")]
    audio_paths = [os.path.join(audio_dir, f.replace(".align", ".wav")) for f in os.listdir(align_dir) if f.endswith(".align")]
    frame_paths = [os.path.join(frame_dir, f.replace(".align", "")) for f in os.listdir(align_dir) if f.endswith(".align")]
    output_dirs = [os.path.join(output_dir, f.replace(".align", "")) for f in os.listdir(align_dir) if f.endswith(".align")]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=50) as executor:
        futures = []
        for align_file, audio_path, v_frame_dir, v_output_dir in zip(align_files, audio_paths, frame_paths, output_dirs):
            os.makedirs(v_output_dir, exist_ok=True)
            futures.append(executor.submit(process_alignment, align_file, audio_path, v_frame_dir, v_output_dir))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing file: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
