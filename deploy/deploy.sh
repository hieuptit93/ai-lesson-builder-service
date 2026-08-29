#!/bin/bash
# Deploy AI Lesson Builder Service
# Usage: ./deploy.sh [local|docker|server]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_MODE="${1:-docker}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_env() {
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_error ".env file not found!"
        log_info "Creating from .env.example..."
        cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        log_warn "Please edit .env with your API keys before deploying"
        exit 1
    fi

    # Check for API key
    if ! grep -q "OPENAI_API_KEY=sk-" "$PROJECT_DIR/.env" && \
       ! grep -q "ANTHROPIC_API_KEY=sk-ant" "$PROJECT_DIR/.env"; then
        log_error "No valid API key found in .env"
        log_warn "Please add OPENAI_API_KEY or ANTHROPIC_API_KEY"
        exit 1
    fi
}

deploy_local() {
    log_info "Deploying locally..."
    cd "$PROJECT_DIR"

    # Create venv if not exists
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3.12 -m venv venv || python3 -m venv venv
    fi

    # Activate and install
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    # Run
    log_info "Starting server..."
    python run.py
}

deploy_docker() {
    log_info "Deploying with Docker..."
    cd "$PROJECT_DIR"

    # Load env vars
    export $(grep -v '^#' .env | xargs)

    # Build and run
    log_info "Building Docker image..."
    docker-compose -f deploy/docker-compose.prod.yml build

    log_info "Starting containers..."
    docker-compose -f deploy/docker-compose.prod.yml up -d

    log_info "Waiting for service to be healthy..."
    sleep 5

    # Check health
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        log_info "✅ Service is running at http://localhost:8000"
        log_info "📖 API Docs: http://localhost:8000/docs"
    else
        log_error "Service failed to start. Check logs:"
        docker-compose -f deploy/docker-compose.prod.yml logs
        exit 1
    fi
}

deploy_server() {
    log_info "Deploying to remote server..."

    # Check for SERVER_HOST
    if [ -z "$SERVER_HOST" ]; then
        log_error "SERVER_HOST not set"
        log_info "Usage: SERVER_HOST=your-server.com ./deploy.sh server"
        exit 1
    fi

    SERVER_USER="${SERVER_USER:-root}"
    SERVER_PATH="${SERVER_PATH:-/opt/ai-lesson-builder}"

    log_info "Deploying to $SERVER_USER@$SERVER_HOST:$SERVER_PATH"

    # Sync files
    rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '.git' \
        --exclude 'server.log' --exclude 'server.pid' \
        "$PROJECT_DIR/" "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"

    # Remote commands
    ssh "$SERVER_USER@$SERVER_HOST" << EOF
        cd $SERVER_PATH

        # Install Docker if needed
        if ! command -v docker &> /dev/null; then
            curl -fsSL https://get.docker.com | sh
        fi

        # Deploy with Docker Compose
        export \$(grep -v '^#' .env | xargs)
        docker-compose -f deploy/docker-compose.prod.yml down || true
        docker-compose -f deploy/docker-compose.prod.yml build
        docker-compose -f deploy/docker-compose.prod.yml up -d

        echo "Checking health..."
        sleep 5
        curl -s http://localhost:8000/health
EOF

    log_info "✅ Deployed to $SERVER_HOST"
}

# Main
log_info "AI Lesson Builder Deployment"
log_info "Mode: $DEPLOY_MODE"
echo ""

check_env

case "$DEPLOY_MODE" in
    local)
        deploy_local
        ;;
    docker)
        deploy_docker
        ;;
    server)
        deploy_server
        ;;
    *)
        log_error "Unknown deploy mode: $DEPLOY_MODE"
        echo "Usage: ./deploy.sh [local|docker|server]"
        exit 1
        ;;
esac
