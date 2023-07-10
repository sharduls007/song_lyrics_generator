import os
import re
import time
import math
import torch
import numpy as np
from torch.utils.data import IterableDataset
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["OMP_NUM_THREADS"] = "8"

# Check the availability of the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA device
else:
    device = torch.device("cpu")  # Use CPU device

# Reading data
album_path = "./data/Albums"
text_data = []
for root, dirs, files in os.walk(album_path):
    for name in files:
        with open(os.path.join(root, name), mode="r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
            # adding [EOS] at the end of each song
            lines = " ".join(lines) + " [EOS]"
            ## adding [SEP] in between verses
            lines = re.sub(r"\n \n", " [SEP]", lines)
            lines = re.sub(r"\n", "", lines)
            text_data.append(lines)
            f.close()

# Splitting data into 80-10-10 data split
train, val = train_test_split(text_data, test_size=0.1, random_state=99)
train, test = train_test_split(train, test_size=0.1, random_state=99)

# Dataset
class TaylorLyricsDataset(IterableDataset):
    """
    A custom IterableDataset implementation.

    This class allows iterating over the provided data by implementing the __iter__ method.
    It inherits from the IterableDataset class.

    Args:
        data (Iterable): The data to be used for iteration.

    Yields:
        Any: Each item from the provided data.
    """
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        for item in self.data:
            # Yield or return each item from the data
            yield item

train_iter = TaylorLyricsDataset(train)
test_iter = TaylorLyricsDataset(test)
val_iter = TaylorLyricsDataset(val)

# Tokenizer
tokenizer = get_tokenizer('subword')
vocab = build_vocab_from_iterator(map(tokenizer, map(lambda x:x.strip(),train_iter)), specials=["[UNK]", "[SEP]", "[EOS]"], max_tokens=15346)
vocab.set_default_index(vocab["[UNK]"])

def data_process(raw_text_iter):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item.strip())), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

train_data = data_process(train_iter)
test_data = data_process(test_iter)
val_data = data_process(val_iter)

def batchify(data, seq_len):
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        seq_len: int, sequence length

    Returns:
        Tensor of shape ``[seq_len, seq_len // bsz]``
    """
    bsz = data.size(0) // seq_len
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

# Set to 5 for keeping
seq_len = 512

train_data = batchify(train_data, seq_len)
val_data = batchify(val_data, seq_len)
test_data = batchify(test_data, seq_len)

bptt = 35
def get_batch(source, i):
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# modelling
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Positional encoding calculation
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'

        # Positional encoding layer
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Word embedding layer
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        # Linear decoder layer
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # Generating embeddings from encoder and normalizing it with model dimensions
        src = self.encoder(src) * math.sqrt(self.d_model)
        # Encoding position in the text embeddings
        src = self.pos_encoder(src)
        # Transformer encoder
        output = self.transformer_encoder(src, src_mask)
        # Linear decoder
        output = self.decoder(output)
        return output

    def generate(self, idx, max_new_tokens):
      for _ in range(max_new_tokens):
        logits = self(idx)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
      return idx

def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

config = {
    "ntoken": len(vocab.get_itos()),
    "d_model": 256,
    "d_hid": 512,
    "nlayers": 6,
    "nhead": 4,
    "dropout": 0.2,
    "lr": 1e-3
}


model = TransformerModel(
    ntoken = config["ntoken"],
    d_model = config["d_model"],
    nhead = config["nhead"],
    d_hid = config["d_hid"],
    nlayers = config["nlayers"],
    dropout = config["dropout"]
    ).to(device)

# training
criterion = nn.CrossEntropyLoss()
lr = config["lr"]  # learning rate
# Define the optimizer using stochastic gradient descent (SGD)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# Define a learning rate scheduler that decreases the learning rate over time
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, config["ntoken"])
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()

        # Clip the gradients to prevent exploding gradients problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            # Calculate perplexity
            ppl = math.exp(cur_loss)

            # Print the training progress and metrics
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model, eval_data):
    model.eval()  # Turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)

            output = model(data)
            output_flat = output.view(-1, config["ntoken"])

            # Compute the loss and accumulate it
            total_loss += seq_len * criterion(output_flat, targets).item()
    # Return the average loss over the evaluation data
    return total_loss / (len(eval_data) - 1)

best_val_loss = float('inf')
epochs = 15

best_model_params_path = os.path.join("./model.pth")

for epoch in range(1, epochs + 1):
  epoch_start_time = time.time()
  train(model)
  val_loss = evaluate(model, val_data)
  val_ppl = math.exp(val_loss)
  elapsed = time.time() - epoch_start_time
  print('-' * 89)
  print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
  print('-' * 89)

  if val_loss < best_val_loss:
    best_val_loss = val_loss
    torch.save(model.state_dict(), best_model_params_path)

  scheduler.step()
  model.load_state_dict(torch.load(best_model_params_path))