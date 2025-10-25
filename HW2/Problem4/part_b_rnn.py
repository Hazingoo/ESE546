import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from part_a_simple import get_text, get_vocab, char_to_index, onehot_encode

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

def create_sequences(text, char_map, seq_length=32):
    """Create training sequences from text - predict next character only"""
    vocab_size = len(char_map)
    sequences = []
    targets = []
    
    for i in range(len(text) - seq_length):
        seq = text[i:i+seq_length]  # Input sequence
        target_char = text[i+seq_length]  # Next character (as per assignment)
        
        seq_onehot = onehot_encode(seq, char_map)
        target_idx = char_map[target_char]  # Single character index
        
        sequences.append(seq_onehot)
        targets.append(target_idx)
    
    return torch.stack(sequences), torch.tensor(targets)

def train_model(model, train_data, val_data, char_map, epochs=50, lr=1e-3):
    """Train the RNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
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
    
    train_seqs, train_targets = train_data
    val_seqs, val_targets = val_data
    
    train_seqs = train_seqs.to(device)
    train_targets = train_targets.to(device)
    val_seqs = val_seqs.to(device)
    val_targets = val_targets.to(device)
    
    batch_size = 64
    num_batches = len(train_seqs) // batch_size
    
    weight_update_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_error = 0
        
        for i in range(0, len(train_seqs), batch_size):
            batch_seqs = train_seqs[i:i+batch_size]
            batch_targets = train_targets[i:i+batch_size]
            
            optimizer.zero_grad()
            
            hidden = model.init_hidden(batch_seqs.size(0)).to(device)
            outputs, hidden = model(batch_seqs, hidden)
            
            last_output = outputs[:, -1, :]
            loss = criterion(last_output, batch_targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            weight_update_count += 1
            epoch_loss += loss.item()
            
            with torch.no_grad():
                _, predicted = torch.max(last_output, 1)
                error = (predicted != batch_targets).float().mean()
                epoch_error += error.item()
            
            # Report metrics every 1000 weight updates 
            if weight_update_count % 1000 == 0:
                # Store current training metrics
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
                    
                    val_losses.append(val_loss.item())
                    val_errors.append(val_error.item())
                    val_losses_by_updates.append(val_loss.item())
                    val_errors_by_updates.append(val_error.item())
                    
                    print(f"Weight Update {weight_update_count}: Train Loss={loss.item():.4f}, Train Error={error.item():.4f}, Val Loss={val_loss.item():.4f}, Val Error={val_error.item():.4f}")
                model.train()
        
        avg_loss = epoch_loss / num_batches
        avg_error = epoch_error / num_batches
        train_losses.append(avg_loss)
        train_errors.append(avg_error)
        
        scheduler.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Train Error={avg_error:.4f}")
    
    return (train_losses, train_errors, val_losses, val_errors, 
            train_losses_by_updates, train_errors_by_updates, 
            val_losses_by_updates, val_errors_by_updates, weight_updates_list)

def plot_results(train_losses, train_errors, val_losses, val_errors, 
                train_losses_by_updates, train_errors_by_updates, 
                val_losses_by_updates, val_errors_by_updates, weight_updates_list):
    """Plot training progress"""
    
    # Plot 1: Loss vs Epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss vs Epochs')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_errors, label='Train Error')
    ax2.plot(val_errors, label='Val Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error')
    ax2.set_title('Training and Validation Error vs Epochs')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('rnn_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Loss vs Weight Updates
    if weight_updates_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(weight_updates_list, train_losses_by_updates, 'b-o', label='Train Loss')
        ax1.plot(weight_updates_list, val_losses_by_updates, 'r-o', label='Val Loss')
        ax1.set_xlabel('Weight Updates')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss vs Weight Updates (Every 1000 updates)')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(weight_updates_list, train_errors_by_updates, 'b-o', label='Train Error')
        ax2.plot(weight_updates_list, val_errors_by_updates, 'r-o', label='Val Error')
        ax2.set_xlabel('Weight Updates')
        ax2.set_ylabel('Error')
        ax2.set_title('Error vs Weight Updates (Every 1000 updates)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('rnn_metrics_vs_updates.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    # Load data
    url = "https://www.gutenberg.org/ebooks/100.txt.utf-8"
    text = get_text(url)
    
    # Had memory issues so had to truncate text
    max_chars = 500000  
    if len(text) > max_chars:
        text = text[:max_chars]
        print(f"Truncated text to {max_chars} characters to prevent memory issues")
    
    chars = get_vocab(text)
    vocab_size = len(chars)
    char_map = char_to_index(chars)
    
    print(f"Text length: {len(text)}")
    print(f"Vocab size: {vocab_size}")
    
    # Create sequences
    sequences, targets = create_sequences(text, char_map, seq_length=32)
    print(f"Created {len(sequences)} sequences")
    
    # Split train/val 
    split_idx = int(0.8 * len(sequences))
    train_data = (sequences[:split_idx], targets[:split_idx])
    val_data = (sequences[split_idx:], targets[split_idx:])
    
    print(f"Train sequences: {len(train_data[0])}")
    print(f"Val sequences: {len(val_data[0])}")
    
    # Create model
    model = CharRNN(vocab_size, hidden_size=128)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train
    results = train_model(model, train_data, val_data, char_map, epochs=50, lr=1e-3)
    train_losses, train_errors, val_losses, val_errors, train_losses_by_updates, train_errors_by_updates, val_losses_by_updates, val_errors_by_updates, weight_updates_list = results
    
    # Plot results
    plot_results(train_losses, train_errors, val_losses, val_errors, 
                train_losses_by_updates, train_errors_by_updates, 
                val_losses_by_updates, val_errors_by_updates, weight_updates_list)
    
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Final validation error: {val_errors[-1]:.4f}")
