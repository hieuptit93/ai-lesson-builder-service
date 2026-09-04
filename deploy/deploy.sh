#!/bin/bash
# ============================================
# AI Lesson Builder Service - Deploy Script
# ============================================
# Usage: ./deploy.sh [local|docker|server|status|logs|stop]
#
# Options:
#   local   - Run locally with Python venv
#   docker  - Deploy with Docker Compose
#   server  - Deploy to remote server via SSH
#   status  - Show service status
#   logs    - Show service logs
#   stop    - Stop the service
#   restart - Restart the service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOY_MODE="${1:-docker}"
SERVICE_NAME="ai-lesson-builder"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

print_banner() {
    echo -e "${GREEN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║        AI Lesson Builder Service - Deployment             ║"
    echo "║                     Version 2.0.0                         ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_env() {
    log_step "Checking environment..."

    if [ ! -f "$PROJECT_DIR/.env" ]; then
        log_error ".env file not found!"
        log_info "Creating from template..."

        if [ -f "$SCRIPT_DIR/.env.production.example" ]; then
            cp "$SCRIPT_DIR/.env.production.example" "$PROJECT_DIR/.env"
        elif [ -f "$PROJECT_DIR/.env.example" ]; then
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
        else
            log_error "No .env template found"
            exit 1
        fi

        log_warn "Please edit .env with your API keys before deploying"
        log_info "Required: OPENAI_API_KEY"
        exit 1
    fi

    # Check for OpenAI API key
    if ! grep -q "OPENAI_API_KEY=sk-" "$PROJECT_DIR/.env"; then
        log_error "OPENAI_API_KEY not found or invalid in .env"
        log_warn "Please add a valid OpenAI API key starting with 'sk-'"
        exit 1
    fi

    log_info "Environment configuration OK"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        log_info "Install Docker: curl -fsSL https://get.docker.com | sh"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    log_info "Docker OK"
}

get_compose_cmd() {
    if docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

deploy_local() {
    log_step "Deploying locally..."
    cd "$PROJECT_DIR"

    # Create venv if not exists
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3.12 -m venv venv || python3 -m venv venv
    fi

    # Activate and install
    log_info "Installing dependencies..."
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q

    # Get port from .env
    PORT=$(grep "^PORT=" "$PROJECT_DIR/.env" | cut -d'=' -f2 || echo "8000")

    # Run
    log_info "Starting server on port $PORT..."
    log_info "API Docs: http://localhost:$PORT/docs"
    python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
}

deploy_docker() {
    log_step "Deploying with Docker..."
    cd "$PROJECT_DIR"

    check_docker

    COMPOSE_CMD=$(get_compose_cmd)
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.prod.yml"

    # Build
    log_info "Building Docker image..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" build

    # Stop existing if running
    log_info "Stopping existing containers..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" down 2>/dev/null || true

    # Start
    log_info "Starting containers..."
    $COMPOSE_CMD -f "$COMPOSE_FILE" up -d

    # Wait and check health
    log_info "Waiting for service to be healthy..."
    sleep 8

    # Get port
    PORT=$(grep "^PORT=" "$PROJECT_DIR/.env" | cut -d'=' -f2 || echo "8000")

    for i in {1..10}; do
        if curl -s "http://localhost:$PORT/health" | grep -q "healthy"; then
            echo ""
            log_info "✅ Service is running!"
            log_info "Health: http://localhost:$PORT/health"
            log_info "API Docs: http://localhost:$PORT/docs"
            return 0
        fi
        echo -n "."
        sleep 2
    done

    echo ""
    log_error "Service failed to start. Logs:"
    $COMPOSE_CMD -f "$COMPOSE_FILE" logs --tail=50
    exit 1
}

deploy_server() {
    log_step "Deploying to remote server..."

    if [ -z "$SERVER_HOST" ]; then
        log_error "SERVER_HOST not set"
        echo ""
        echo "Usage:"
        echo "  SERVER_HOST=your-server.com ./deploy.sh server"
        echo ""
        echo "Optional variables:"
        echo "  SERVER_USER=root (default)"
        echo "  SERVER_PATH=/opt/ai-lesson-builder (default)"
        exit 1
    fi

    SERVER_USER="${SERVER_USER:-root}"
    SERVER_PATH="${SERVER_PATH:-/opt/ai-lesson-builder}"

    log_info "Target: $SERVER_USER@$SERVER_HOST:$SERVER_PATH"

    # Sync files
    log_info "Syncing files..."
    rsync -avz --progress \
        --exclude 'venv' \
        --exclude '__pycache__' \
        --exclude '.git' \
        --exclude 'server.log' \
        --exclude 'server.pid' \
        --exclude '.pytest_cache' \
        --exclude '*.pyc' \
        "$PROJECT_DIR/" "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/"

    # Remote deploy
    log_info "Running remote deploy..."
    ssh "$SERVER_USER@$SERVER_HOST" << 'REMOTE_SCRIPT'
        set -e
        cd $SERVER_PATH

        # Install Docker if needed
        if ! command -v docker &> /dev/null; then
            echo "Installing Docker..."
            curl -fsSL https://get.docker.com | sh
            systemctl enable docker
            systemctl start docker
        fi

        # Deploy
        echo "Building and starting containers..."

        if docker compose version &> /dev/null 2>&1; then
            docker compose -f deploy/docker-compose.prod.yml down || true
            docker compose -f deploy/docker-compose.prod.yml up -d --build
        else
            docker-compose -f deploy/docker-compose.prod.yml down || true
            docker-compose -f deploy/docker-compose.prod.yml up -d --build
        fi

        # Health check
        echo "Checking health..."
        sleep 5
        curl -s http://localhost:8000/health || echo "Warning: Health check failed"
REMOTE_SCRIPT

    log_info "✅ Deployed to $SERVER_HOST"
}

show_status() {
    log_step "Service Status"
    cd "$PROJECT_DIR"

    COMPOSE_CMD=$(get_compose_cmd)
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.prod.yml"

    if command -v docker &> /dev/null; then
        echo ""
        log_info "Docker Containers:"
        $COMPOSE_CMD -f "$COMPOSE_FILE" ps 2>/dev/null || docker ps --filter "name=$SERVICE_NAME"

        echo ""
        PORT=$(grep "^PORT=" "$PROJECT_DIR/.env" | cut -d'=' -f2 || echo "8000")
        log_info "Health Check:"
        curl -s "http://localhost:$PORT/health" 2>/dev/null | python3 -m json.tool || echo "Service not responding"
    fi
}

show_logs() {
    log_step "Service Logs"
    cd "$PROJECT_DIR"

    COMPOSE_CMD=$(get_compose_cmd)
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.prod.yml"

    $COMPOSE_CMD -f "$COMPOSE_FILE" logs -f --tail=100
}

stop_service() {
    log_step "Stopping service..."
    cd "$PROJECT_DIR"

    COMPOSE_CMD=$(get_compose_cmd)
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.prod.yml"

    $COMPOSE_CMD -f "$COMPOSE_FILE" down
    log_info "Service stopped"
}

restart_service() {
    log_step "Restarting service..."
    cd "$PROJECT_DIR"

    COMPOSE_CMD=$(get_compose_cmd)
    COMPOSE_FILE="$SCRIPT_DIR/docker-compose.prod.yml"

    $COMPOSE_CMD -f "$COMPOSE_FILE" restart

    sleep 5
    PORT=$(grep "^PORT=" "$PROJECT_DIR/.env" | cut -d'=' -f2 || echo "8000")
    curl -s "http://localhost:$PORT/health" && echo ""
    log_info "Service restarted"
}

# Main
print_banner
echo "Mode: $DEPLOY_MODE"
echo ""

case "$DEPLOY_MODE" in
    local)
        check_env
        deploy_local
        ;;
    docker)
        check_env
        deploy_docker
        ;;
    server)
        check_env
        deploy_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service
        ;;
    *)
        echo "Usage: ./deploy.sh [command]"
        echo ""
        echo "Commands:"
        echo "  local    - Run locally with Python venv"
        echo "  docker   - Deploy with Docker Compose (default)"
        echo "  server   - Deploy to remote server via SSH"
        echo "  status   - Show service status"
        echo "  logs     - Show service logs (follow)"
        echo "  stop     - Stop the service"
        echo "  restart  - Restart the service"
        echo ""
        echo "Environment variables for 'server' command:"
        echo "  SERVER_HOST  - Remote server hostname (required)"
        echo "  SERVER_USER  - SSH user (default: root)"
        echo "  SERVER_PATH  - Deploy path (default: /opt/ai-lesson-builder)"
        exit 1
        ;;
esac
