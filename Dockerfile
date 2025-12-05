# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_lg

# Copy all Python files
COPY bigram_model.py .
COPY lstm_model.py .
COPY gpt2_model.py .
COPY word_embedding.py .
COPY main_embedding.py .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000
