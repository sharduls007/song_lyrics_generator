import time
import os
import math
import torch
import torch.nn as nn
from config import config
from model import TransformerModel
from data import train_data, val_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


if __name__ == "__main__":
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    model = TransformerModel(
        ntoken=config["ntoken"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        d_hid=config["d_hid"],
        nlayers=config["nlayers"],
        dropout=config["dropout"]
    ).to(device)

    # Training
    criterion = nn.CrossEntropyLoss()
    lr = config["lr"]  # learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

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