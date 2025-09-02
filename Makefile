# Advanced RAG System Makefile

.PHONY: help setup install clean start-db stop-db start-api start-web test lint format

# Default target
help:
	@echo "Advanced RAG System Commands"
	@echo "============================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup      - Initial setup of the project"
	@echo "  make install    - Install Python dependencies"
	@echo ""
	@echo "Database Commands:"
	@echo "  make start-db   - Start vector database (Docker)"
	@echo "  make stop-db    - Stop vector database"
	@echo "  make db-status  - Check database status"
	@echo ""
	@echo "Application Commands:"
	@echo "  make start-api  - Start FastAPI server"
	@echo "  make start-web  - Start Streamlit web interface"
	@echo "  make start-all  - Start database and web interface"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code"
	@echo ""
	@echo "Utility Commands:"
	@echo "  make clean      - Clean up temporary files"
	@echo "  make reset      - Reset the entire system"

# Setup commands
setup:
	@echo "🚀 Setting up Advanced RAG System..."
	@chmod +x setup.sh
	@./setup.sh
	@echo "✅ Setup complete! Edit .env with your API keys."

install:
	@echo "📦 Installing Python dependencies..."
	@pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Database commands
start-db:
	@echo "🐳 Starting vector database..."
	@docker-compose up -d
	@echo "✅ Database started! Qdrant available at http://localhost:6333"

stop-db:
	@echo "🛑 Stopping vector database..."
	@docker-compose down
	@echo "✅ Database stopped!"

db-status:
	@echo "🔍 Checking database status..."
	@docker-compose ps
	@echo ""
	@curl -s http://localhost:6333/collections 2>/dev/null && echo "✅ Qdrant is responding" || echo "❌ Qdrant is not responding"

# Application commands
start-api:
	@echo "🚀 Starting FastAPI server..."
	@uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

start-web:
	@echo "🌐 Starting Streamlit web interface..."
	@streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

start-all: start-db
	@echo "🚀 Starting all services..."
	@sleep 5  # Wait for database to be ready
	@echo "Starting web interface..."
	@streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Development commands
test:
	@echo "🧪 Running tests..."
	@python -m pytest tests/ -v --cov=. --cov-report=html
	@echo "✅ Tests completed! Coverage report in htmlcov/"

lint:
	@echo "🔍 Running code linting..."
	@flake8 . --max-line-length=88 --exclude=venv,qdrant_storage
	@black --check .
	@isort --check-only .
	@echo "✅ Linting completed!"

format:
	@echo "🎨 Formatting code..."
	@black .
	@isort .
	@echo "✅ Code formatted!"

# Utility commands
clean:
	@echo "🧹 Cleaning up temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name "htmlcov" -exec rm -rf {} +
	@find . -type f -name ".coverage" -delete
	@echo "✅ Cleanup completed!"

reset: clean stop-db
	@echo "🔄 Resetting system..."
	@rm -rf qdrant_storage/
	@mkdir -p qdrant_storage
	@echo "⚠️  System reset! You'll need to re-upload documents."

# Development helpers
dev-setup:
	@echo "🛠️ Setting up development environment..."
	@pip install -r requirements.txt
	@pip install pytest pytest-cov flake8 black isort
	@pre-commit install
	@echo "✅ Development environment ready!"

check-env:
	@echo "🔍 Checking environment..."
	@python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('OpenAI API Key:', 'Set' if os.getenv('OPENAI_API_KEY') else 'Not Set')"
	@echo ""
	@python -c "import sys; print('Python Version:', sys.version)"
	@docker --version 2>/dev/null && echo "Docker: Available" || echo "Docker: Not Available"

demo:
	@echo "🎯 Running demo setup..."
	@echo "This will setup the system with sample documents"
	@make start-db
	@sleep 5
	@python -c "from advanced_rag import RAGSystem, RAGConfig; rag = RAGSystem(); rag.setup_system('./documents/samples'); print('Demo setup complete!')"
	@echo "✅ Demo ready! Run 'make start-web' to try it out."

# Docker commands for advanced users
docker-build:
	@echo "🐳 Building Docker image..."
	@docker build -t advanced-rag:latest .

docker-run:
	@echo "🚀 Running in Docker..."
	@docker run -p 8501:8501 -p 8000:8000 -v $(PWD)/documents:/app/documents advanced-rag:latest

# Backup and restore
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@tar -czf backups/rag-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz documents/ qdrant_storage/ .env
	@echo "✅ Backup created in backups/"

restore:
	@echo "📁 Available backups:"
	@ls -la backups/
	@echo "Use: tar -xzf backups/your-backup.tar.gz"

# Performance testing
perf-test:
	@echo "⚡ Running performance tests..."
	@python scripts/performance_test.py
	@echo "✅ Performance test completed!"

# Monitoring
logs:
	@echo "📋 Showing recent logs..."
	@docker-compose logs -f

status:
	@echo "📊 System Status"
	@echo "==============="
	@make db-status
	@echo ""
	@curl -s http://localhost:8000/status 2>/dev/null || echo "API: Not running"
	@echo ""
	@curl -s http://localhost:8501 2>/dev/null && echo "Web: Running" || echo "Web: Not running"