# Use an official slim Python image
FROM python:3.9-slim

# Install system dependencies including Tesseract OCR and build tools
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model automatically
RUN python -m spacy download en_core_web_sm

# Copy the entire project into the container
COPY ../EntityExtraction .

# Alternatively, if your .dockerignore is excluding Dataset,
# explicitly copy it:
 COPY Dataset /app/Dataset

# Run your pipeline script
CMD ["python", "pipeline.py"]
