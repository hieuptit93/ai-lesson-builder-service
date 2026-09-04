#!/bin/bash
# ============================================
# Quick Start Script
# ============================================
# One-command setup for fresh deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "🚀 AI Lesson Builder - Quick Start"
echo "==================================="
echo ""

# Check if .env exists
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo "📝 Creating .env from template..."
    cp "$SCRIPT_DIR/.env.production.example" "$PROJECT_DIR/.env"

    echo ""
    echo "⚠️  Please configure your .env file:"
    echo "   nano $PROJECT_DIR/.env"
    echo ""
    echo "Required settings:"
    echo "   - OPENAI_API_KEY=sk-proj-xxx"
    echo ""
    echo "Optional but recommended:"
    echo "   - PROFILE_API_BASE_URL"
    echo "   - MEM0_BASE_URL"
    echo "   - LANGFUSE_SECRET_KEY"
    echo "   - GOOGLE_CHAT_WEBHOOK_URL"
    echo ""
    echo "After editing .env, run: ./deploy/deploy.sh docker"
    exit 0
fi

# Check for API key
if ! grep -q "OPENAI_API_KEY=sk-" "$PROJECT_DIR/.env"; then
    echo "❌ OPENAI_API_KEY not configured in .env"
    echo "   Edit: nano $PROJECT_DIR/.env"
    exit 1
fi

echo "✅ Configuration found"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "📦 Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "⚠️  Please logout and login again, then re-run this script"
    exit 1
fi

echo "✅ Docker installed"
echo ""

# Deploy
echo "🐳 Deploying with Docker..."
"$SCRIPT_DIR/deploy.sh" docker

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📖 Documentation:"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Health: http://localhost:8000/health"
echo "   - Deploy Guide: $SCRIPT_DIR/DEPLOY.md"
echo ""
echo "🔧 Useful commands:"
echo "   ./deploy/deploy.sh logs     # View logs"
echo "   ./deploy/deploy.sh status   # Check status"
echo "   ./deploy/deploy.sh restart  # Restart service"
echo "   ./deploy/deploy.sh stop     # Stop service"
