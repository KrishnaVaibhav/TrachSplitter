FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create volumes for input/output
VOLUME ["/input", "/output"]

# Set environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Create non-root user
RUN useradd -m -u 1000 trachsplitter && \
    chown -R trachsplitter:trachsplitter /app /input /output

USER trachsplitter

# Default command
CMD ["python", "main.py"]

# Usage:
# docker build -t trachsplitter .
# docker run --gpus all -v /path/to/audio:/input -v /path/to/output:/output trachsplitter
# Or for CLI:
# docker run --gpus all -v /path/to/audio:/input -v /path/to/output:/output trachsplitter python cli.py separate /input/song.mp3 -o /output
