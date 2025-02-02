from src.data.vocab import Vocab
from src.data.data_loader import load_vocab_from_json
from src.data.dataset import LipReadingDataset
import json 
import torch
import torch.nn.functional as F
# data_dir = "data/processed/s1"
# vocab_path= "data/raw/word_to_idx.json"
# vocab = load_vocab_from_json("data/raw/word_to_idx.json")
# dataset = LipReadingDataset(data_dir=data_dir, vocab=vocab)

# # with open(vocab_path, 'r') as f:
# #     word2idx = json.load(f)
# # specials = {'pad': '<pad>', 'unk': '<unk>', 'sos': '<sos>', 'eos': '<eos>'}
# # vocab = Vocab(tokens=list(word2idx.keys()), specials=specials)
# print("Vocabulary mapping (stoi):", vocab.stoi)

# print("token_to_id('bin') =", vocab.token_to_id('bin'))
# print("token_to_id('blue') =", vocab.token_to_id('blue'))
# print("token_to_id('at') =", vocab.token_to_id('at'))

# for i in range(5):
#     frames, label_tensor = dataset[i]  # label_tensor is a tensor of indices
#     print(f"Sample {i}: frames shape = {frames.shape}")
#     print("Label tensor:", label_tensor)
    
#     # Convert token IDs to a Python list
#     token_ids = label_tensor.tolist()
#     print("Token IDs:", token_ids)
    
#     # Convert token IDs to their corresponding strings
#     token_strings = [vocab.id_to_token(token_id) for token_id in token_ids]
#     print("Decoded tokens:", token_strings)


ctc_log_probs = torch.randn(75, 2, 57, requires_grad=True).log_softmax(2)
labels = torch.randint(1, 57, (2, 75), dtype=torch.long)
frame_lengths = torch.tensor([75, 75])
label_lengths = torch.tensor([10, 10])  # Shorter than frame_lengths
loss_ctc = F.ctc_loss(ctc_log_probs, labels, frame_lengths, label_lengths)
print("CTC Loss (synthetic):", loss_ctc.item())
loss_ctc.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(name, param.grad.sum())