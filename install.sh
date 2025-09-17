#!/bin/bash

# Installation script for BERT training on research papers
# This script sets up the environment and dependencies

echo "🤖 Setting up BERT Training Environment"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if Python 3.8+ is installed
if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    echo "❌ Python 3.8+ is required"
    exit 1
fi

echo "✅ Python version check passed"

# Install pip dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Download NLTK data
echo "📚 Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'❌ Failed to download NLTK data: {e}')
    exit(1)
"

# Check if papers are downloaded
if [ -d "downloaded_papers/train" ] && [ "$(ls -A downloaded_papers/train/pdfs/ 2>/dev/null)" ]; then
    train_count=$(ls downloaded_papers/train/pdfs/*.pdf 2>/dev/null | wc -l)
    echo "✅ Found $train_count training papers"
else
    echo "⚠️  No training papers found"
    echo "   Run: python download_papers.py --separate-folders"
    echo "   to download the research papers dataset"
fi

# Create output directories
echo "📁 Creating output directories..."
mkdir -p bert_training_output/{processed_data,models,logs}
mkdir -p evaluation_results

# Check GPU availability
echo "🖥️  Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠️  No GPU detected - training will use CPU (slower)')
"

echo ""
echo "🎉 Installation completed!"
echo ""
echo "Next steps:"
echo "1. Download papers (if not done): python download_papers.py --separate-folders"
echo "2. Quick test: python quick_start.py --mode quick"
echo "3. View examples: python example_usage.py"
echo "4. Start MLflow UI: python quick_start.py --mode ui"
echo ""
echo "Happy training! 🚀"

