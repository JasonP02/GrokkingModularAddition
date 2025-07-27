from black.trans import Transformer
# imports
import torch
import torch.nn as nn

class OneLayerTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, vocab_size, seq_len):
        super().__init__()
        self.embedding = nn.Embedding()
