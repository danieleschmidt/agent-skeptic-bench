#!/bin/bash

# Development container setup script for Agent Skeptic Bench

echo "🚀 Setting up Agent Skeptic Bench development environment..."

# Update system packages
sudo apt-get update
sudo apt-get install -y curl wget git

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p data

# Set up environment template
if [ ! -f .env ]; then
    echo "📝 Creating environment template..."
    cat > .env << EOF
# AI Provider API Keys (add your keys here)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_AI_API_KEY=your_google_key_here

# Development settings
AGENT_SKEPTIC_DEBUG=true
LOG_LEVEL=INFO
PYTEST_CURRENT_TEST=true

# Optional: Custom model endpoints
# CUSTOM_MODEL_ENDPOINT=http://localhost:8000
EOF
    echo "⚠️  Please update .env with your API keys before running evaluations"
fi

# Verify installation
echo "🧪 Running verification tests..."
python -c "import agent_skeptic_bench; print('✅ Package import successful')"

# Run quick test suite
pytest tests/unit/ -q --tb=no --disable-warnings
if [ $? -eq 0 ]; then
    echo "✅ Unit tests passed"
else
    echo "❌ Some unit tests failed - check your setup"
fi

# Set up git hooks
echo "🔗 Configuring git hooks..."
git config --local commit.template .gitmessage 2>/dev/null || true

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Update .env with your API keys"
echo "  2. Run 'pytest' to run the full test suite"
echo "  3. Run 'mkdocs serve' to start the documentation server"
echo "  4. Start coding! 🚀"
echo ""