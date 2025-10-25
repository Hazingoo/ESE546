from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from part_a_simple import get_text, get_vocab, char_to_index, onehot_encode

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # RNN layer that processes sequences
        self.rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
        # Linear layer to convert hidden state to vocabulary predictions
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # Process the input sequence through RNN
        rnn_out, hidden = self.rnn(x, hidden)
        # Convert hidden states to vocabulary predictions
        output = self.fc(rnn_out)
        return output, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state to zeros
        return torch.zeros(1, batch_size, self.hidden_size)

def create_sequences(text, char_map, seq_length=32):
    sequences = []
    targets = []
    
    # Create training sequences by sliding window
    for i in range(len(text) - seq_length):
        seq = text[i:i+seq_length]  # Input sequence
        target_char = text[i+seq_length]  # Next character to predict
        
        seq_onehot = onehot_encode(seq, char_map)
        target_idx = char_map[target_char]
        
        sequences.append(seq_onehot)
        targets.append(target_idx)
    
    return torch.stack(sequences), torch.tensor(targets)

if __name__ == '__main__':
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Load text data from Project Gutenberg
    url = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
    text = get_text(url)
    
    # Truncate text to prevent memory issues
    max_chars = 300000
    if len(text) > max_chars:
        text = text[:max_chars]
        print('Truncated text to', max_chars, 'characters')
    
    # Build vocabulary from text
    chars = get_vocab(text)
    vocab_size = len(chars)
    char_map = char_to_index(chars)
    
    print('Text length:', len(text))
    print('Vocab size:', vocab_size)
    
    # Create training sequences
    sequences, targets = create_sequences(text, char_map, seq_length=32)
    print('Created', len(sequences), 'sequences')
    
    # Split into train and validation sets
    split_idx = int(0.8 * len(sequences))
    train_data = (sequences[:split_idx], targets[:split_idx])
    val_data = (sequences[split_idx:], targets[split_idx:])
    
    print('Train sequences:', len(train_data[0]))
    print('Val sequences:', len(val_data[1]))
    
    # Create the RNN model
    model = CharRNN(vocab_size, hidden_size=128)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Set up loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Move data to device
    train_seqs, train_targets = train_data
    val_seqs, val_targets = val_data
    
    train_seqs = train_seqs.to(device)
    train_targets = train_targets.to(device)
    val_seqs = val_seqs.to(device)
    val_targets = val_targets.to(device)
    
    # Training setup
    batch_size = 64
    num_batches = len(train_seqs) // batch_size
    
    # Lists to store training metrics
    train_losses = []
    train_errors = []
    val_losses = []
    val_errors = []
    
    # Track metrics vs weight updates 
    train_losses_by_updates = []
    train_errors_by_updates = []
    val_losses_by_updates = []
    val_errors_by_updates = []
    weight_updates_list = []
    weight_update_count = 0
    
    # Training loop
    for epoch in range(50):
        print('EPOCH:', epoch)
        model.train()
        epoch_loss = 0
        epoch_error = 0
        
        # Process batches
        for i in range(0, len(train_seqs), batch_size):
            batch_seqs = train_seqs[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            optimizer.zero_grad()
            
            # Forward pass through RNN
            hidden = model.init_hidden(batch_seqs.size(0)).to(device)
            outputs, hidden = model(batch_seqs, hidden)
            
            # Use only the last timestep output for next character prediction
            last_output = outputs[:, -1, :]
            loss = criterion(last_output, batch_targets)
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            weight_update_count += 1
            epoch_loss += loss.item()
            
            # Calculate training error
            with torch.no_grad():
                _, predicted = torch.max(last_output, 1)
                error = (predicted != batch_targets).float().mean()
                epoch_error += error.item()
            
            # Report metrics every 1000 weight updates 
            if weight_update_count % 1000 == 0:
                train_losses_by_updates.append(loss.item())
                train_errors_by_updates.append(error.item())
                weight_updates_list.append(weight_update_count)
                
                # Calculate validation metrics
                model.eval()
                with torch.no_grad():
                    val_hidden = model.init_hidden(val_seqs.size(0)).to(device)
                    val_outputs, _ = model(val_seqs, val_hidden)
                    val_last_output = val_outputs[:, -1, :]
                    
                    val_loss = criterion(val_last_output, val_targets)
                    _, val_predicted = torch.max(val_last_output, 1)
                    val_error = (val_predicted != val_targets).float().mean()
                    
                    val_losses_by_updates.append(val_loss.item())
                    val_errors_by_updates.append(val_error.item())
                    
                    print('Weight Update', weight_update_count, ': Train Loss =', loss.item(), ', Train Error =', error.item(), ', Val Loss =', val_loss.item(), ', Val Error =', val_error.item())
                model.train()
        
        # Calculate average metrics for this epoch
        avg_loss = epoch_loss / num_batches
        avg_error = epoch_error / num_batches
        train_losses.append(avg_loss)
        train_errors.append(avg_error)
        
        # Update learning rate
        scheduler.step()
        
        # Validation at end of each epoch
        with torch.no_grad():
            model.eval()
            val_hidden = model.init_hidden(val_seqs.size(0)).to(device)
            val_outputs, _ = model(val_seqs, val_hidden)
            val_last_output = val_outputs[:, -1, :]
            
            val_loss = criterion(val_last_output, val_targets)
            _, val_predicted = torch.max(val_last_output, 1)
            val_error = (val_predicted != val_targets).float().mean()
            
            val_losses.append(val_loss.item())
            val_errors.append(val_error.item())
            
            print('train loss:', avg_loss)
            print('train error:', avg_error)
            print('val loss:', val_loss.item())
            print('val error:', val_error.item())
    
    # Plot training progress vs epochs
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'r', label='Train Loss')
    plt.plot(val_losses, 'b', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_errors, 'r', label='Train Error')
    plt.plot(val_errors, 'b', label='Val Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Training and Validation Error vs Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_training_progress.pdf')
    plt.close()
    
    # Plot metrics vs weight updates 
    if weight_updates_list:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(weight_updates_list, train_losses_by_updates, 'r-o', label='Train Loss')
        plt.plot(weight_updates_list, val_losses_by_updates, 'b-o', label='Val Loss')
        plt.xlabel('Weight Updates')
        plt.ylabel('Loss')
        plt.title('Loss vs Weight Updates (Every 1000 updates)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(weight_updates_list, train_errors_by_updates, 'r-o', label='Train Error')
        plt.plot(weight_updates_list, val_errors_by_updates, 'b-o', label='Val Error')
        plt.xlabel('Weight Updates')
        plt.ylabel('Error')
        plt.title('Error vs Weight Updates (Every 1000 updates)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('rnn_metrics_vs_updates.pdf')
        plt.close()
    
    # Print final results
    print('Final validation loss:', val_losses[-1])
    print('Final validation error:', val_errors[-1])
