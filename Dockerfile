# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY app/ ./app/
COPY main_embedding.py .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main_embedding:app", "--host", "0.0.0.0", "--port", "8000"]