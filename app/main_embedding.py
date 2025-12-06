from typing import Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .bigram_model import BigramModel
from .word_embedding import WordEmbedding
from .lstm_model import LSTMTextGenerator
from .gpt2_model import GPT2TextGenerator
import os

app = FastAPI(
    title="Text Generation API",
    description="Multi-model text generation API with Bigram, LSTM, and fine-tuned GPT-2",
    version="3.0.0"
)

# Initialize models
print("Initializing models...")

# Word embedding model
nlp_model = WordEmbedding()

# Bigram model (Module 3)
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

# LSTM model (Module 7) - load only if checkpoint exists
lstm_model = None
LSTM_MODEL_PATH = '../models/lstm_checkpoint.pth'

try:
    if os.path.exists(LSTM_MODEL_PATH):
        lstm_model = LSTMTextGenerator(LSTM_MODEL_PATH)
        print("✓ LSTM model loaded successfully!")
    else:
        print(f"⚠ LSTM model not found at {LSTM_MODEL_PATH}")
        print("  Run 'python train_lstm.py' to train the model.")
except Exception as e:
    print(f"✗ Error loading LSTM model: {e}")

# GPT-2 model (Assignment 5) - load only if fine-tuned model exists
gpt2_model = None
GPT2_MODEL_PATH = '../models/gpt2_finetuned'

try:
    if os.path.exists(GPT2_MODEL_PATH):
        gpt2_model = GPT2TextGenerator(GPT2_MODEL_PATH)
        print("✓ GPT-2 model loaded successfully!")
    else:
        print(f"⚠ GPT-2 model not found at {GPT2_MODEL_PATH}")
        print("  Run 'python train_gpt2.py' to fine-tune the model.")
except Exception as e:
    print(f"✗ Error loading GPT-2 model: {e}")

print("\nApplication initialized!")
print(f"Models loaded: Bigram ✓ | LSTM {'✓' if lstm_model else '✗'} | GPT-2 {'✓' if gpt2_model else '✗'}")


# Request models
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class QuestionRequest(BaseModel):
    question: str
    max_length: int = 50

class WordRequest(BaseModel):
    word: str


# Endpoints
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Model Text Generation API",
        "version": "3.0.0",
        "endpoints": {
            "/generate": "Generate text using Bigram model (Module 3)",
            "/generate_with_rnn": "Generate text using LSTM model (Module 7)",
            "/generate_with_gpt2": "Answer questions using fine-tuned GPT-2 (Assignment 5)",
            "/embedding": "Get word embedding"
        },
        "models_loaded": {
            "bigram": True,
            "lstm": lstm_model is not None,
            "gpt2": gpt2_model is not None,
            "embeddings": True
        },
        "instructions": {
            "lstm": "Run 'python train_lstm.py' if not loaded",
            "gpt2": "Run 'python train_gpt2.py' if not loaded"
        }
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Generate text using the Bigram model (Module 3)
    
    Args:
        start_word: The word to start generation from
        length: Number of words to generate
    
    Returns:
        Generated text
    """
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
        return {
            "model": "Bigram",
            "module": "Module 3",
            "generated_text": generated_text,
            "start_word": request.start_word,
            "length": request.length
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    """
    Generate text using the LSTM model (Module 7)
    
    Args:
        start_word: The text to start generation from (can be multiple words)
        length: Number of words to generate
    
    Returns:
        Generated text
    """
    if lstm_model is None:
        raise HTTPException(
            status_code=503,
            detail="LSTM model not available. Please run 'python train_lstm.py' to train the model first."
        )
    
    try:
        generated_text = lstm_model.generate_text(
            seed_text=request.start_word,
            length=request.length,
            temperature=1.0
        )
        
        return {
            "model": "LSTM",
            "module": "Module 7",
            "generated_text": generated_text,
            "seed_text": request.start_word,
            "length": request.length
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")


@app.post("/generate_with_gpt2")
def generate_with_gpt2(request: QuestionRequest):
    """
    Answer questions using fine-tuned GPT-2 (Assignment 5)
    
    The model has been fine-tuned to answer in the format:
    "That is a great question. [ANSWER]. Let me know if you have any other questions!"
    
    Args:
        question: The question to answer
        max_length: Maximum length of the generated answer (default: 150)
    
    Returns:
        Generated answer in the trained format
    """
    if gpt2_model is None:
        raise HTTPException(
            status_code=503,
            detail="GPT-2 model not available. Please run 'python train_gpt2.py' to fine-tune the model first."
        )
    
    try:
        answer = gpt2_model.generate_answer(
            question=request.question,
            max_length=request.max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        return {
            "model": "GPT-2 (Fine-tuned)",
            "module": "Assignment 5",
            "question": request.question,
            "answer": answer,
            "format": "That is a great question. [ANSWER]. Let me know if you have any other questions!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/embedding")
def calculate_embedding(request: WordRequest):
    """
    Calculate word embedding for a given word
    
    Args:
        word: The word to get embedding for
    
    Returns:
        First 10 dimensions of the embedding vector
    """
    try:
        embedding = nlp_model.calculate_embedding(request.word)
        return {
            "word": request.word,
            "embedding_preview": embedding[:10],
            "embedding_dimension": len(embedding)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating embedding: {str(e)}")


@app.get("/health")
def health_check():
    """Check if all models are loaded and ready"""
    return {
        "status": "healthy",
        "models": {
            "bigram": "ready",
            "lstm": "ready" if lstm_model is not None else "not_loaded",
            "gpt2": "ready" if gpt2_model is not None else "not_loaded",
            "embeddings": "ready"
        },
        "assignment_progress": {
            "module_3": "✓ Complete (Bigram)",
            "module_7": "✓ Complete (LSTM)" if lstm_model is not None else "✗ Pending",
            "assignment_5": "✓ Complete (GPT-2)" if gpt2_model is not None else "✗ Pending"
        }
    }


@app.get("/models/compare")
def compare_models():
    """
    Get information about all available models and their capabilities
    """
    return {
        "available_models": [
            {
                "name": "Bigram",
                "module": "Module 3",
                "endpoint": "/generate",
                "status": "ready",
                "description": "Simple statistical model based on word pairs",
                "pros": ["Fast", "No training needed", "Simple"],
                "cons": ["Only considers previous word", "Less coherent"]
            },
            {
                "name": "LSTM",
                "module": "Module 7",
                "endpoint": "/generate_with_rnn",
                "status": "ready" if lstm_model is not None else "not_loaded",
                "description": "Neural network trained on 'The Count of Monte Cristo'",
                "pros": ["Considers 30-word context", "More coherent", "Learns patterns"],
                "cons": ["Requires training", "Slower than bigram"]
            },
            {
                "name": "GPT-2 (Fine-tuned)",
                "module": "Assignment 5",
                "endpoint": "/generate_with_gpt2",
                "status": "ready" if gpt2_model is not None else "not_loaded",
                "description": "Pre-trained GPT-2 fine-tuned on SQuAD with custom format",
                "pros": ["Best quality", "Follows specific format", "Pre-trained knowledge"],
                "cons": ["Requires fine-tuning", "Largest model"]
            }
        ]
    }
