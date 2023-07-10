from data import vocab

config = {
    "ntoken": len(vocab.get_itos()),
    "d_model": 256,
    "d_hid": 512,
    "nlayers": 6,
    "nhead": 4,
    "dropout": 0.2,
    "lr": 1e-3
}