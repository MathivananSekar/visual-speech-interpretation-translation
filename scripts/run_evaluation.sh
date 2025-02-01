python -m src.inference.lip_reading --video_path data/raw/s1/videos/bbaf2n.mpg --checkpoint experiments/checkpoints/lipreading_transformer_epoch6.pt --vocab_json data/raw/s1/word_to_idx.json


python -m src.inference.validate_videos --video_dir  data/raw/s1/videos --align_dir data/raw/s1/alignments --checkpoint experiments/checkpoints/lipreading_transformer_epoch6.pt --vocab_json data/raw/s1/word_to_idx.json