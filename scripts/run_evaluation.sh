
python -m src.inference.lip_reading --video_path data/raw/s1/videos/bbaz6p.mpg --checkpoint experiments/checkpoints/c2/lipreading_transformer_epoch6.pt --vocab_json data/raw/word_to_idx.json

python -m src.inference.validate_videos --video_dir  data/raw/s1/videos --align_dir data/raw/s1/alignments --checkpoint experiments/checkpoints/c2/lipreading_transformer_epoch6.pt --vocab_json data/raw/word_to_idx.json

python -m src.training.validation_with_metrics