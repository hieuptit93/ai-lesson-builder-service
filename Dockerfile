# AI Lesson Builder Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application.
# --timeout-keep-alive must stay ABOVE the idle timeout of every pooling client
# in front of us (nginx upstream keepalive: 60s). Uvicorn's 5s default made the
# server drop idle connections while clients still held them pooled, so the next
# request was written into a dying socket - surfacing on Reactor Netty callers as
# "Connection prematurely closed BEFORE response".
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "75"]
