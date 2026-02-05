# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root)
RUN useradd -m -u 1000 -s /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY Requirements.txt .
RUN pip install --no-cache-dir -r Requirements.txt

# Copy application code
COPY app/ ./app/
COPY config.yaml .

# Create directories for runtime data
RUN mkdir -p cache runs store inbox logs && \
    chown -R app:app /app

# Switch to app user
USER app

# Default command (can be overridden)
CMD ["python", "-m", "app.main", "--help"]
