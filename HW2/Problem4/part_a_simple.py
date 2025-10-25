import requests
import torch

def get_text(url):
    response = requests.get(url)
    return response.text

def get_vocab(text):
    # Extract unique characters from text 
    chars = set(text)  # Get unique characters
    return sorted(chars)  # Return sorted list

def char_to_index(chars):
    # Create mapping from characters to indices 
    return {char: i for i, char in enumerate(chars)}

def onehot_encode(seq, char_map):
    # Convert character sequence to one-hot encoding 
    vocab_size = len(char_map)
    seq_len = len(seq)
    onehot = torch.zeros(seq_len, vocab_size)  # Initialize tensor
    
    for i, char in enumerate(seq):
        if char in char_map:
            onehot[i, char_map[char]] = 1.0  # Set corresponding index to 1
    
    return onehot

if __name__ == '__main__':
    # Download Project Gutenberg text
    url = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
    text = get_text(url)
    
    # Build vocabulary
    chars = get_vocab(text)
    print("vocab size:", len(chars))
    
    # Create character to index mapping
    char_map = char_to_index(chars)
    
    # Take first 32 characters and encode
    seq = text[:32]
    onehot = onehot_encode(seq, char_map)
    
    print("sequence:", seq)
    print("shape:", onehot.shape)
