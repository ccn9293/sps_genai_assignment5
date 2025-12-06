import torch
import torch.nn as nn
import pickle
import os


class LSTMModel(nn.Module):
    """LSTM-based text generation model"""
    
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


class LSTMTextGenerator:
    """Wrapper class for LSTM text generation with vocabulary management"""
    
    def __init__(self, model_path='models/lstm_checkpoint.pth'):
        """
        Initialize the LSTM text generator
        
        Args:
            model_path: Path to the saved model checkpoint
        """
        self.device = torch.device('cpu')
        
        # Loadoing the checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please run train_lstm.py first to train and save the model."
            )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Loading vocabulary
        self.vocab = checkpoint['vocab']
        self.inv_vocab = checkpoint['inv_vocab']
        
        # Initializing and load model
        self.model = LSTMModel(
            vocab_size=len(self.vocab),
            embedding_dim=100,
            hidden_dim=128
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
    
    def generate_text(self, seed_text, length=50, temperature=1.0):
        """
        Generate text starting from seed_text
        
        Args:
            seed_text: Starting text (string)
            length: Number of words to generate
            temperature: Controls randomness (higher = more random)
            
        Returns:
            Generated text as string
        """
        self.model.eval()
        
        # Tokenize seed text
        words = seed_text.lower().split()
        input_ids = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        
        
        if not input_ids:
            input_ids = [2]  
        
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        hidden = None
        
        with torch.no_grad():
            for _ in range(length):
                # Get prediction
                output, hidden = self.model(input_tensor, hidden)
                
                # Apply temperature and get probabilities
                logits = output[0, -1] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Sample next token
                next_id = torch.multinomial(probs, num_samples=1).item()
                words.append(self.inv_vocab.get(next_id, "<UNK>"))
                
                # Prepare next input
                input_ids.append(next_id)
                input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        
        return " ".join(words)


# For backward compatibility and easy testing
def load_lstm_model(model_path='models/lstm_checkpoint.pth'):
    """
    Convenience function to load the LSTM model
    
    Args:
        model_path: Path to saved model checkpoint
        
    Returns:
        LSTMTextGenerator instance
    """
    return LSTMTextGenerator(model_path)
