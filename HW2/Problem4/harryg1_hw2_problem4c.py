from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from harryg1_hw2_problem4a import get_text, get_vocab, char_to_index, onehot_encode

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn(x, hidden)
        output = self.fc(rnn_out)
        return output, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

def generate_text(model, char_map, index_to_char, seed_text, length=200):
    model.eval()
    device = next(model.parameters()).device
    
    # Convert seed text to one-hot
    seed_onehot = onehot_encode(seed_text, char_map).unsqueeze(0).to(device)
    
    # Initialize hidden state to zero
    hidden = model.init_hidden(1).to(device)
    
    # Process seed text to get initial hidden state
    with torch.no_grad():
        for i in range(seed_onehot.size(1)):
            char_input = seed_onehot[:, i:i+1, :]
            _, hidden = model(char_input, hidden)
    
    # Generate new characters
    generated_text = seed_text
    current_char = seed_onehot[:, -1:, :]  # Last character of seed
    
    with torch.no_grad():
        for _ in range(length):
            # Get prediction
            output, hidden = model(current_char, hidden)
            
            # Convert to probabilities
            probs = torch.softmax(output, dim=-1)
            
            # Sample next character
            char_idx = torch.multinomial(probs.squeeze(), 1).item()
            
            # Convert to one-hot for next iteration
            current_char = torch.zeros(1, 1, len(char_map)).to(device)
            current_char[0, 0, char_idx] = 1.0
            
            # Add to generated text
            generated_text += index_to_char[char_idx]
    
    return generated_text

if __name__ == '__main__':
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Load data
    url = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
    text = get_text(url)
    
    # Truncate text
    max_chars = 300000
    if len(text) > max_chars:
        text = text[:max_chars]
        print('Truncated text to', max_chars, 'characters')
    
    # Build vocabulary
    chars = get_vocab(text)
    vocab_size = len(chars)
    char_map = char_to_index(chars)
    index_to_char = {i: char for char, i in char_map.items()}
    
    print('Text length:', len(text))
    print('Vocab size:', vocab_size)
    
    # Load trained model
    model = CharRNN(vocab_size, hidden_size=128)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load('harryg1_hw2_problem4b.pth', map_location=device))
        model.eval()
        print("Loaded trained model")
    except FileNotFoundError:
        print("Trained model not found, using untrained model")
        model.eval()
    
    # Generate text examples
    print("GENERATED TEXT EXAMPLES\n")
    
    # Different seed texts
    seeds = [
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "Hello world",
        "To be or not to be"
    ]
    
    for i, seed in enumerate(seeds, 1):
        print(f"\nExample {i} (Seed: '{seed}'):")
        print("-" * 40)
        
        generated = generate_text(model, char_map, index_to_char, seed, length=150)
        print(f"Generated: {generated}")
        print()
    
