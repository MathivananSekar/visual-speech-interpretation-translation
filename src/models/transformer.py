import torch
import torch.nn as nn
from src.models.backbones import SpatioTemporalResNet
from src.models.positional_encoding import PositionalEncoding
    
class LipReading3DTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=4,
                 num_decoder_layers=4, dim_feedforward=1024, max_len=250, 
                 dropout=0.1, use_ctc=False):
        super().__init__()
        self.visual_backbone = SpatioTemporalResNet(return_sequence=True)
        self.feature_dim = 512
        self.visual_fc = nn.Linear(self.feature_dim, d_model)
        
        self.pos_encoder_vis = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder_txt = PositionalEncoding(d_model, dropout, max_len, batch_first=True)
        self.output_fc = nn.Linear(d_model, vocab_size)
        
        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_vocab_size = vocab_size + 1
            self.ctc_fc = nn.Linear(d_model, self.ctc_vocab_size)
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=1.41)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def encode_video(self, video_frames):
        feats = self.visual_backbone(video_frames)
        feats = self.visual_fc(feats)
        feats = self.pos_encoder_vis(feats)
        memory = self.transformer.encoder(feats)
        return memory

    def decode_text(self, tgt_tokens, memory, pad_id=0):
        emb = self.token_embedding(tgt_tokens)
        emb = self.pos_encoder_txt(emb)
        L = tgt_tokens.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L).to(tgt_tokens.device)
        tgt_key_padding_mask = (tgt_tokens == pad_id)
        output = self.transformer.decoder(
            emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        logits = self.output_fc(output)
        return logits

    def forward(self, video_frames, tgt_tokens=None, input_lengths=None, target_lengths=None, pad_id=0):
        memory = self.encode_video(video_frames)
        outputs = {}
        if tgt_tokens is not None:
            seq2seq_logits = self.decode_text(tgt_tokens, memory, pad_id)
            outputs['seq2seq_logits'] = seq2seq_logits
        if self.use_ctc:
            ctc_logits = self.ctc_fc(memory)
            outputs['ctc_logits'] = ctc_logits
        return outputs