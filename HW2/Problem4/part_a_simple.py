import requests
import torch

def get_text(url):
    response = requests.get(url)
    return response.text

def get_vocab(text):
    chars = set(text)
    m = len(chars)
    return sorted(chars), m

def char_to_index(chars):
    return {char: i for i, char in enumerate(chars)}

def onehot_encode(seq, char_map):
    vocab_size = len(char_map)
    seq_len = len(seq)
    onehot = torch.zeros(seq_len, vocab_size)
    
    for i, char in enumerate(seq):
        if char in char_map:
            onehot[i, char_map[char]] = 1.0
    
    return onehot

if __name__ == '__main__':
    url = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
    text = get_text(url)
    
    chars, m = get_vocab(text)
    print("vocab size:", m)
    
    char_map = char_to_index(chars)
    
    seq = text[:32]
    onehot = onehot_encode(seq, char_map)
    
    print("sequence:", seq)
    print("shape:", onehot.shape)
    print("sum check:", torch.all(torch.sum(onehot, dim=1) == 1))
