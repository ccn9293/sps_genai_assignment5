import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import requests
import re
from collections import Counter
import os
from app.lstm_model import LSTMModel

SEQ_LEN = 30
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS = 15
MODEL_SAVE_PATH = 'models/lstm_checkpoint.pth'


class TextDataset(Dataset):
    """Dataset for sequential text data"""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data) - SEQ_LEN
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+SEQ_LEN]),
            torch.tensor(self.data[idx+1:idx+SEQ_LEN+1])
        )


def load_and_preprocess_text():
    """Load Count of Monte Cristo and preprocess it"""
    print("Loading text from Project Gutenberg...")
    
    url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
    text = requests.get(url).text
    
    # Extract main body
    start_idx = text.find("Chapter 1.")
    end_idx = text.rfind("Chapter 5.")
    text = text[start_idx:end_idx]
    
    # Preprocessing
    print("Preprocessing text...")
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    
    return text


def build_vocabulary(tokens, vocab_size=10000):
    """Build vocabulary from tokens"""
    print("Building vocabulary...")
    
    counter = Counter(tokens)
    
    # Create vocabulary with special tokens
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(vocab_size-2))}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    
    # Create inverse vocabulary
    inv_vocab = {idx: word for word, idx in vocab.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    
    return vocab, inv_vocab


def encode_text(tokens, vocab):
    """Encode tokens to indices"""
    print("Encoding text...")
    return [vocab.get(word, vocab["<UNK>"]) for word in tokens]


def train_model():
    """Main training function"""
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load and prepare data
    text = load_and_preprocess_text()
    tokens = text.split()
    print(f"Total tokens: {len(tokens)}")
    
    # Build vocabulary
    vocab, inv_vocab = build_vocabulary(tokens, VOCAB_SIZE)
    
    # Encode text
    encoded = encode_text(tokens, vocab)
    
    # Create dataset and dataloader
    print("Creating dataset...")
    train_dataset = TextDataset(encoded)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training samples: {len(train_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    device = torch.device('cpu')
    model = LSTMModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print("This may take a while on CPU...\n")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    
    # Save model, vocabulary, and configuration
    print(f"\nSaving model to {MODEL_SAVE_PATH}...")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'inv_vocab': inv_vocab,
        'config': {
            'vocab_size': len(vocab),
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'seq_len': SEQ_LEN
        }
    }
    
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print("Model saved successfully!")
    
    # Test generation
    print("\n" + "="*50)
    print("Testing text generation...")
    print("="*50)
    
    from app.lstm_model import LSTMTextGenerator
    
    generator = LSTMTextGenerator(MODEL_SAVE_PATH)
    
    test_seeds = [
        "the count of monte cristo",
        "it was a dark",
        "the ship"
    ]
    
    for seed in test_seeds:
        print(f"\nSeed: '{seed}'")
        generated = generator.generate_text(seed, length=30)
        print(f"Generated: {generated}")
    
    print("\n" + "="*50)
    print("Training complete! You can now use the model in your API.")
    print("="*50)


if __name__ == "__main__":
    print("="*50)
    print("LSTM Text Generation Model Training")
    print("="*50)
    print("\nThis script will:")
    print("1. Download 'The Count of Monte Cristo' from Project Gutenberg")
    print("2. Preprocess the text")
    print("3. Build a vocabulary")
    print("4. Train an LSTM model (this may take 15-30 minutes on CPU)")
    print("5. Save the trained model to 'models/lstm_checkpoint.pth'")
    print("\n" + "="*50 + "\n")
    
    train_model()