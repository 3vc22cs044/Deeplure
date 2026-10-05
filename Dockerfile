FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements-cpu.txt .

# Install PyTorch CPU and Python dependencies
RUN pip install --no-cache-dir -r requirements-cpu.txt

# Copy application files
COPY . .

# Expose port (7860 for Hugging Face Spaces, default for containers)
EXPOSE 7860

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
