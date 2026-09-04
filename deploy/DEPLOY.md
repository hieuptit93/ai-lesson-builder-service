# AI Lesson Builder Service - Deployment Guide

## Table of Contents
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Clone and configure
cd /opt
git clone https://github.com/hieuptit93/ai-lesson-builder-service.git
cd ai-lesson-builder-service

# 2. Setup environment
cp deploy/.env.production.example .env
nano .env  # Edit with your API keys

# 3. Deploy with Docker
./deploy/deploy.sh docker

# 4. Verify
curl http://localhost:8000/health
```

---

## Prerequisites

### Server Requirements
| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 2 GB | 4+ GB |
| Disk | 10 GB | 20 GB |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |

### Required Services
- **Docker** 20.10+ & Docker Compose v2
- **Python** 3.12+ (for local deployment)

### API Keys Required
| Service | Purpose | Get from |
|---------|---------|----------|
| OpenAI | Vision & Lesson generation | https://platform.openai.com |
| Langfuse | LLM Observability (optional) | https://cloud.langfuse.com |
| Profile API | User profiles | Internal |
| Mem0 | Memory/RAG | Internal |

---

## Deployment Options

### Option 1: Docker (Recommended)

```bash
# Build and start
./deploy/deploy.sh docker

# Or manually:
docker-compose -f deploy/docker-compose.prod.yml up -d --build

# Check logs
docker logs -f ai-lesson-builder

# Stop
docker-compose -f deploy/docker-compose.prod.yml down
```

### Option 2: Docker with Nginx (SSL)

```bash
# Generate SSL certs first
mkdir -p deploy/ssl
# Place your cert.pem and key.pem in deploy/ssl/

# Deploy with nginx profile
docker-compose -f deploy/docker-compose.prod.yml --profile with-nginx up -d
```

### Option 3: Local Development

```bash
./deploy/deploy.sh local

# Or manually:
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

### Option 4: Remote Server

```bash
# Set server details
export SERVER_HOST=your-server.com
export SERVER_USER=root
export SERVER_PATH=/opt/ai-lesson-builder

# Deploy
./deploy/deploy.sh server
```

### Option 5: Systemd Service (No Docker)

```bash
# Create service file
sudo tee /etc/systemd/system/ai-lesson-builder.service << EOF
[Unit]
Description=AI Lesson Builder Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/ai-lesson-builder-service
Environment=PATH=/opt/ai-lesson-builder-service/venv/bin
EnvironmentFile=/opt/ai-lesson-builder-service/.env
ExecStart=/opt/ai-lesson-builder-service/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable ai-lesson-builder
sudo systemctl start ai-lesson-builder
sudo systemctl status ai-lesson-builder
```

---

## Configuration

### Environment Variables

#### Core Settings
```env
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

#### OpenAI Models
```env
# Vision (image extraction) - requires vision capability
OPENAI_VISION_MODEL=gpt-5.6-terra      # Best: $2/M in, $12/M out
OPENAI_SUGGESTIONS_MODEL=gpt-5.6-terra  # Same for v3/lessons/generate

# Guardrail (safety check) - cheap model
OPENAI_GUARDRAIL_MODEL=gpt-5.6-luna     # $0.20/M in

# Lesson generation
OPENAI_LESSON_MODEL=gpt-4.1             # $2/M in, $8/M out
```

#### Performance Tuning
```env
# Parallel image processing (v3)
OPENAI_V3_PARALLEL_IMAGES=true
OPENAI_V3_IMAGES_PER_BATCH=1   # Images per API call
OPENAI_V3_MAX_CONCURRENT=4     # Max parallel calls

# Token limits
OPENAI_VISION_MAX_TOKENS=16000
OPENAI_LESSON_MAX_TOKENS=8000
```

#### Alerting
```env
GOOGLE_CHAT_WEBHOOK_URL=https://chat.googleapis.com/v1/spaces/xxx/messages?key=xxx&token=xxx
ALERTING_ENABLED=true
```

---

## Monitoring

### Health Check
```bash
curl http://localhost:8000/health
# {"status":"healthy","service":"ai-lesson-builder-service","version":"2.0.0"}
```

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Logs
```bash
# Docker
docker logs -f ai-lesson-builder

# Systemd
journalctl -u ai-lesson-builder -f

# Local
tail -f server.log
```

### Langfuse Dashboard
Access your Langfuse instance to view:
- Request traces
- Token usage
- Latency metrics
- Error rates

### Google Chat Alerts
Alerts are sent for:
- Profile API errors (HIGH severity)
- LLM timeouts (MEDIUM severity)
- Rate limits (HIGH severity)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/lessons/generate` | POST | Generate full lessons (5-expert, ~10s) |
| `/api/v3/lessons/generate` | POST | Generate suggested lessons (~11s) |
| `/api/v3/lessons/generate_artifact` | POST | Generate lesson artifacts (~26s) |

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/lessons/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "profile_id": "user123",
    "image_urls": ["https://example.com/worksheet.jpg"],
    "optional_parent_config": {
      "subject": "english",
      "purpose": "review",
      "language": "vi"
    }
  }'
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker logs ai-lesson-builder

# Common issues:
# - Missing API key: Check .env file
# - Port in use: Change PORT in .env
# - Memory limit: Increase in docker-compose.yml
```

### High latency
```bash
# Check model selection
grep OPENAI_.*_MODEL .env

# Verify parallel processing is enabled
grep V3_PARALLEL .env

# Expected latencies:
# - v1/generate: ~10s (single call)
# - v3/generate: ~11s (parallel images)
# - v3/artifact: ~26s (parallel lessons)
```

### Alerts not sending
```bash
# Test webhook manually
curl -X POST "YOUR_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test alert"}'

# Check firewall allows outbound HTTPS to chat.googleapis.com
```

### Memory issues
```bash
# Increase Docker memory limit
# In docker-compose.prod.yml:
deploy:
  resources:
    limits:
      memory: 8G
```

---

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.prod.yml
services:
  ai-lesson-builder:
    deploy:
      replicas: 3
```

### Load Balancing with Nginx
```nginx
upstream ai_backend {
    server ai-lesson-builder-1:8000;
    server ai-lesson-builder-2:8000;
    server ai-lesson-builder-3:8000;
}
```

---

## Security Checklist

- [ ] Change default Swagger password
- [ ] Enable X-API-Key authentication
- [ ] Use HTTPS in production
- [ ] Rotate API keys regularly
- [ ] Review Langfuse access permissions
- [ ] Secure .env file permissions (`chmod 600 .env`)

---

## Support

- GitHub: https://github.com/hieuptit93/ai-lesson-builder-service
- Issues: Create a GitHub issue for bugs/features
