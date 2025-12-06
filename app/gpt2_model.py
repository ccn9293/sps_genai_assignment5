import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2TextGenerator:
    """Wrapper class for fine-tuned GPT-2 text generation"""
    
    def __init__(self, model_path='models/gpt2_finetuned'):
        """
        Initialize the GPT-2 text generator
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.device = torch.device('cpu')
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}. "
                "Please run train_gpt2.py first to fine-tune the model."
            )
        
        print(f"Loading GPT-2 model from {model_path}...")
        
        # Load tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("GPT-2 model loaded successfully!")
    
    def generate_answer(self, question, max_length=150, temperature=0.7, top_p=0.9):
        """
        Generate an answer to a question in the trained format
        
        Args:
            question: The question to answer (string)
            max_length: Maximum length of generated text
            temperature: Controls randomness (higher = more random)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated answer as string
        """
        self.model.eval()
        
       
        prompt = f"Question: {question}\nAnswer:"
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate response
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3  # Avoid repetition
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract just the answer part
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:", 1)[1].strip()
        else:
            answer = generated_text.strip()
        
        return answer
    
    def generate_text(self, prompt, max_length=100, temperature=0.7):
        """
        Alternative method for free-form text generation
        
        Args:
            prompt: Starting text
            max_length: Maximum length of generated text
            temperature: Controls randomness
            
        Returns:
            Generated text as string
        """
        self.model.eval()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=2
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return generated_text


def load_gpt2_model(model_path='models/gpt2_finetuned'):
    """
    Convenience function to load the fine-tuned GPT-2 model
    
    Args:
        model_path: Path to fine-tuned model directory
        
    Returns:
        GPT2TextGenerator instance
    """
    return GPT2TextGenerator(model_path)


# test
if __name__ == "__main__":
    print("Testing GPT-2 Text Generator...")
    
    try:
        generator = GPT2TextGenerator()
        
        test_questions = [
            "Who wrote Romeo and Juliet?",
            "What is the capital of France?",
            "How does photosynthesis work?"
        ]
        
        print("\nGenerating answers...")
        for question in test_questions:
            print(f"\nQ: {question}")
            answer = generator.generate_answer(question)
            print(f"A: {answer}")
            
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run train_gpt2.py first to fine-tune the model.")