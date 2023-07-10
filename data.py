import os
import re
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# Custom dataset
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

# Reading data
def read_data(album_path = "./data/Albums"):
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
    return text_data

def data_process(raw_text_iter):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item.strip())), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

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
    return data

album_path = "./data/Albums"
text_data = read_data(album_path)
# Splitting data into 80-10-10 data split
train, val = train_test_split(text_data, test_size=0.1, random_state=99)
train, test = train_test_split(train, test_size=0.1, random_state=99)

train_iter = TaylorLyricsDataset(train)
test_iter = TaylorLyricsDataset(test)
val_iter = TaylorLyricsDataset(val)

# Tokenizer
tokenizer = get_tokenizer('subword')
vocab = build_vocab_from_iterator(map(tokenizer, map(lambda x:x.strip(),train_iter)), specials=["[UNK]", "[SEP]", "[EOS]"], max_tokens=15346)
vocab.set_default_index(vocab["[UNK]"])

# Data pre-processing
train_data = data_process(train_iter)
test_data = data_process(test_iter)
val_data = data_process(val_iter)

# Set to 5 for keeping
seq_len = 512
train_data = batchify(train_data, seq_len)
val_data = batchify(val_data, seq_len)
test_data = batchify(test_data, seq_len)



