
import os
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np


# Configuration
MODEL_NAME = "openai-community/gpt2"  
OUTPUT_DIR = "models/gpt2_finetuned"
MAX_LENGTH = 256
BATCH_SIZE = 4  
EPOCHS = 3  
LEARNING_RATE = 5e-5


def format_qa_pair(example):
    """
    Format a question-answer pair with our custom template
    
    Args:
        example: A single example from SQuAD dataset
        
    Returns:
        Formatted string with our template
    """
    question = example['question']
    
    
    if example['answers']['text']:
        answer = example['answers']['text'][0]
    else:
        return None  # Skip examples without answers
    

    formatted = (
        f"Question: {question}\n"
        f"Answer: That is a great question. {answer}. "
        f"Let me know if you have any other questions!"
    )
    
    return formatted


def prepare_dataset(tokenizer, num_examples=1000):
    """
    Load and prepare the SQuAD dataset
    
    Args:
        tokenizer: GPT-2 tokenizer
        num_examples: Number of examples to use (limited for faster training)
        
    Returns:
        Tokenized dataset
    """
    print("Loading SQuAD dataset...")
    
    # Loading datasest
    dataset = load_dataset("rajpurkar/squad", split="train")
    
    # Take a subset for faster training, otherwise too big
    dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    print(f"Using {len(dataset)} examples from SQuAD")
    
    # Format the examples
    print("Formatting examples with custom template...")
    
    formatted_texts = []
    for example in dataset:
        formatted = format_qa_pair(example)
        if formatted: 
            formatted_texts.append(formatted)
    
    print(f"Successfully formatted {len(formatted_texts)} examples")
    
    # Show a sample
    print("\n" + "="*60)
    print("Sample formatted example:")
    print("="*60)
    print(formatted_texts[0])
    print("="*60 + "\n")
    
    # Tokenize
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples,
            truncation=True,
            max_length=MAX_LENGTH,
            padding='max_length',
            return_tensors='pt'
        )
    
    # Tokenize all texts
    tokenized_data = tokenize_function(formatted_texts)
    
    # Create a dataset dictionary
    tokenized_dataset = {
        'input_ids': tokenized_data['input_ids'],
        'attention_mask': tokenized_data['attention_mask'],
    }
    
    return tokenized_dataset, formatted_texts


def train_gpt2():
    """Main training function"""
    
    print("="*60)
    print("GPT-2 Fine-tuning for Question Answering")
    print("Assignment 5 - Post-Training an LLM")
    print("="*60)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load pre-trained GPT-2 model and tokenizer
    print(f"Loading base GPT-2 model: {MODEL_NAME}")
    print("This may take a moment to download (~500MB)...")
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    print("Model loaded successfully!")
    print(f"Model size: {model.num_parameters():,} parameters")
    
    # Prepare dataset
    tokenized_dataset, formatted_texts = prepare_dataset(tokenizer, num_examples=1000)
    
    # Create a simple Dataset class
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings['input_ids'])
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'labels': self.encodings['input_ids'][idx]  # For language modeling
            }
    
    train_dataset = SimpleDataset(tokenized_dataset)
    
    print(f"\nDataset prepared: {len(train_dataset)} examples")
    
    # Training arguments
    print("\nSetting up training configuration...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  
    )
    
    # Initialize Trainer
    print("Initializing trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting fine-tuning...")
    print("="*60)
    print(f"Training for {EPOCHS} epochs")
    print(f"Batch size: {BATCH_SIZE}")
    print("This will take 1-2 hours on CPU...")
    print("You can do other things while this runs!")
    print("="*60 + "\n")
    
    trainer.train()
    
    # Save the fine-tuned model
    print("\n" + "="*60)
    print("Saving fine-tuned model...")
    print("="*60)
    
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"Model saved to: {OUTPUT_DIR}")
    
    # Test the model
    print("\n" + "="*60)
    print("Testing the fine-tuned model...")
    print("="*60)
    
    test_questions = [
        "Who wrote Romeo and Juliet?",
        "What is the capital of France?",
        "When did World War II end?"
    ]
    
    model.eval()
    device = torch.device('cpu')
    model.to(device)
    
    for question in test_questions:
        prompt = f"Question: {question}\nAnswer:"
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=150,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=3
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        if "Answer:" in generated_text:
            answer = generated_text.split("Answer:", 1)[1].strip()
        else:
            answer = generated_text.strip()
        
        print(f"\nQ: {question}")
        print(f"A: {answer}")
    
    print("\n" + "="*60)
    print("Fine-tuning complete!")
    print("="*60)
    print(f"\nYour fine-tuned model is saved in: {OUTPUT_DIR}")
    print("\nYou can now:")
    print("1. Start your API: uvicorn main_embedding:app --reload")
    print("2. Test the /generate_with_gpt2 endpoint")
    print("3. Commit your code to GitHub")
    print("\n" + "="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ASSIGNMENT 5: Fine-tuning GPT-2")
    print("="*60)
    print("\nThis script will:")
    print("1. Download the base GPT-2 model (~500MB)")
    print("2. Load the SQuAD question-answer dataset")
    print("3. Format answers with: 'That is a great question...'")
    print("4. Fine-tune GPT-2 (1-2 hours on CPU)")
    print("5. Save the fine-tuned model")
    print("\n" + "="*60 + "\n")
    
    input("Press Enter to start fine-tuning (or Ctrl+C to cancel)...")
    print()
    
    train_gpt2()