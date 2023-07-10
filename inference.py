import os
import torch
from torchtext.data.utils import get_tokenizer 
from data import test, vocab, tokenizer
from model import TransformerModel
from config import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(prompt, max_seq_len, temperature, model, tokenizer, vocab, device=device, beam_width=4, seed=0):
    # Set the model to evaluation mode
    model.eval()
    # Set the random seed if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Tokenize the prompt and convert to indices using the vocabulary
    tokens = tokenizer(prompt.strip())
    prompt_indices = [vocab[t] for t in tokens]

    with torch.no_grad():
        beam = [(prompt_indices, 0.0)] # Initialize the beam with the prompt
        completed_sequences = [] # Store completed sequences

        for _ in range(max_seq_len):
            candidates = [] # Store candidate sequences for the next step

            # Expand the beam by generating new candidates
            for seq_indices, seq_score in beam:
                input_tensor = torch.LongTensor(seq_indices).unsqueeze(1).to(device)
                output = model(input_tensor)
                logits = output[-1, -1, :]

                probs = torch.softmax(logits / temperature, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, beam_width)

                # Generate new candidates based on top-k probabilities
                for prob, index in zip(topk_probs.squeeze(), topk_indices.squeeze()):
                    new_seq_indices = seq_indices + [index.item()]
                    new_seq_score = seq_score - torch.log(prob).item()
                    candidates.append((new_seq_indices, new_seq_score))

            # Select top-k candidates for the next iteration
            candidates = sorted(candidates, key=lambda x: x[1])[:beam_width]
            beam = []

            # Check if any candidates have completed sequences
            for candidate_indices, candidate_score in candidates:
                if candidate_indices[-1] == vocab["[EOS]"]:
                    completed_sequences.append((candidate_indices, candidate_score))
                else:
                    beam.append((candidate_indices, candidate_score))

            # Break the loop if enough completed sequences have been found
            if len(completed_sequences) >= beam_width:
                break
    try:
      # Select the best completed sequence with the lowest score
      best_sequence_indices, _ = min(completed_sequences, key=lambda x: x[1])
    except ValueError:
      # If no completed sequences testare found, select the best candidate from the beam
      best_sequence_indices, _ = min(beam, key=lambda x: x[1])

    # Convert the sequence indices back to tokens
    itos = vocab.get_itos()
    generated_text = [itos[i] for i in best_sequence_indices]
    return generated_text

if __name__ == "__main__":

    model_path = "./model.pth"
    model = TransformerModel(
        ntoken = config["ntoken"],
        d_model = config["d_model"],
        nhead = config["nhead"],
        d_hid = config["d_hid"],
        nlayers = config["nlayers"],
        dropout = config["dropout"]
        ).to(device)

    with torch.no_grad():
        model.load_state_dict(torch.load(model_path))

    prompt = test[0][:70]
    max_seq_len = 50

    temperatures = [0.12, 0.7, 0.75, 0.8, 1.0]
    for temperature in temperatures:
        generation = generate_text(prompt, max_seq_len, temperature, model, tokenizer, vocab)
        print(str(temperature)+'\n'+' '.join(generation)+'\n')