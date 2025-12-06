# Assignment 5

Fine-tuned GPT-2 on the SquaD dataset


# Download Pre-trained Models via releases

Models are not included in this repository due to file size

**Download models.zip:** [GitHub Releases](https://github.com/ccn9293/sps_genai_assignment5/releases/tag/v1.0-models) (~444MB)

# Run with docker

```bash
# Clone repo
git clone https://github.com/ccn9293/sps_genai_assignment5.git
cd sps_genai_assignment5

# Download and extract models.zip from the release link above
unzip models.zip

# Build & run
docker build -t text-gen-api .
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models:ro text-gen-api

# Access API
open http://localhost:8000/docs
```

# Run without docker

```bash
# Install dependencies
pip install -e .
python -m spacy download en_core_web_lg

# Download and extract models.zip

# Run API
uvicorn app.main_embedding:app --reload
```

# API Endpoints

- **POST /generate** - Bigram model
- **POST /generate_with_rnn** - LSTM model  
- **POST /generate_with_gpt2** - Fine-tuned GPT-2

Documentation: http://localhost:8000/docs

# Folder structure

```
sps_genai_assignment5/
├── app/
│   ├── main_embedding.py
│   ├── bigram_model.py
│   ├── lstm_model.py
│   ├── gpt2_model.py
│   └── word_embedding.py
├── models/              
│   ├── lstm_checkpoint.pth
│   └── gpt2_finetuned/
├── train_lstm.py
├── train_gpt2.py
├── pyproject.toml
└── Dockerfile
```

# Models

- Bigram model from assignment 1
- LSTM: from assignment 5
- GPT-2: Fine-tuned on SQuAD dataset as suggested (took 2 hours of training)

# Dependencies
Main dependencies:
- FastAPI, PyTorch, Transformers, spaCy
