import shutil
import os

print("Hello")

frame_path = "../data/datasets/s1/frames/bbaf2n/frame_0000.jpg"
output_path = "../data/datasets/s1/labeled_data/bbaf2n/"
# Ensure the output path exists
os.makedirs(output_path, exist_ok=True)

# Copy the frame file to the output path
shutil.copy(frame_path, os.path.join(output_path, os.path.basename(frame_path)))

print(f"File copied from {frame_path} to {output_path}")
