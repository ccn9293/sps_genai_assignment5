# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Download spacy model
RUN python -m spacy download en_core_web_lg

# Copy app folder with all Python files
COPY app/ ./app/

# Create models directory (will be mounted as volume)
RUN mkdir -p models

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main_embedding:app", "--host", "0.0.0.0", "--port", "8000"]
