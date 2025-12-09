# Dockerfile — full VM stack (Streamlit UI + backend deps)
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# System libs needed for PDF/OCR and typical packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl unzip poppler-utils \
    tesseract-ocr libtesseract-dev pkg-config \
    libxml2 libxslt1.1 libjpeg-dev libpng-dev \
    ca-certificates gcc g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
# Using requirements.txt in repo root
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit UI port (8501) and backend port (8000 only for internal container network)
EXPOSE 8501 8000

# Ensure start.sh exists and is executable
RUN chmod +x /app/start.sh || true

# Default command: start.sh handles DB setup/ingest then starts Streamlit
CMD ["/app/start.sh"]
