#!/usr/bin/env python3
"""
Quick start script for BERT training on research papers.

This script provides an easy way to get started with different training scenarios:
1. Quick test mode (small dataset, 1 epoch)
2. Development mode (medium dataset, 2 epochs)
3. Full training mode (complete dataset, 3+ epochs)
4. Hyperparameter search mode
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'torch', 'transformers', 'datasets', 'mlflow', 
        'PyPDF2', 'pdfplumber', 'nltk', 'sklearn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.info("Please install dependencies with: pip install -r requirements.txt")
        return False
    
    # Test Hugging Face model access
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_auth_token=False)
        logger.info("✅ Hugging Face model access successful")
    except Exception as e:
        logger.warning(f"⚠️ Hugging Face model access issue: {e}")
        logger.info("🔧 Run: python3 setup_huggingface.py to fix authentication")
        
        choice = input("Continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            return False
    
    return True


def check_data():
    """Check if downloaded papers are available."""
    base_dir = Path('downloaded_papers')
    
    if not base_dir.exists():
        logger.error("Downloaded papers directory not found!")
        logger.info("Please run download_papers.py first to download the dataset.")
        return False
    
    train_dir = base_dir / 'train'
    if not train_dir.exists() or not list(train_dir.glob('pdfs/*.pdf')):
        logger.error("Training papers not found!")
        logger.info("Please ensure papers are downloaded in downloaded_papers/train/")
        return False
    
    train_count = len(list((train_dir / 'pdfs').glob('*.pdf')))
    logger.info(f"Found {train_count} training papers")
    
    return True


def run_quick_test():
    """Run quick test mode."""
    logger.info("🚀 Starting Quick Test Mode")
    logger.info("- Using 1000 training samples")
    logger.info("- 1 epoch")
    logger.info("- Small batch size")
    
    cmd = [
        'python3', 'train_bert_papers.py',
        '--quick_test',
        '--experiment_name', 'quick_test',
        '--epochs', '1',
        '--max_train_samples', '1000',
        '--batch_size', '8'
    ]
    
    return subprocess.run(cmd)


def run_development():
    """Run development mode."""
    logger.info("🔧 Starting Development Mode")
    logger.info("- Using 5000 training samples")
    logger.info("- 2 epochs")
    logger.info("- Medium batch size")
    
    cmd = [
        'python3', 'train_bert_papers.py',
        '--experiment_name', 'development',
        '--epochs', '2',
        '--max_train_samples', '5000',
        '--batch_size', '16'
    ]
    
    return subprocess.run(cmd)


def run_full_training():
    """Run full training mode."""
    logger.info("🎯 Starting Full Training Mode")
    logger.info("- Using all training data")
    logger.info("- 3 epochs")
    logger.info("- Optimal batch size")
    
    cmd = [
        'python3', 'train_bert_papers.py',
        '--experiment_name', 'full_training',
        '--epochs', '3',
        '--batch_size', '16'
    ]
    
    return subprocess.run(cmd)


def run_hyperparameter_search():
    """Run hyperparameter search mode."""
    logger.info("🔍 Starting Hyperparameter Search Mode")
    logger.info("- Grid search over learning rates and batch sizes")
    logger.info("- Quick runs (1 epoch each)")
    logger.info("- Final training with best parameters")
    
    cmd = [
        'python3', 'train_bert_papers.py',
        '--run_hyperparameter_search',
        '--experiment_name', 'hyperparameter_search',
        '--max_train_samples', '3000'  # Smaller dataset for HP search
    ]
    
    return subprocess.run(cmd)


def run_custom_training(args):
    """Run custom training with user-specified parameters."""
    logger.info("⚙️  Starting Custom Training Mode")
    
    cmd = ['python3', 'train_bert_papers.py']
    
    # Add custom arguments
    if args.model_name:
        cmd.extend(['--model_name', args.model_name])
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        cmd.extend(['--batch_size', str(args.batch_size)])
    if args.learning_rate:
        cmd.extend(['--learning_rate', str(args.learning_rate)])
    if args.experiment_name:
        cmd.extend(['--experiment_name', args.experiment_name])
    if args.max_samples:
        cmd.extend(['--max_train_samples', str(args.max_samples)])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd)


def start_mlflow_ui():
    """Start MLflow UI."""
    logger.info("🖥️  Starting MLflow UI...")
    logger.info("MLflow UI will be available at: http://localhost:5000")
    
    try:
        subprocess.Popen(['mlflow', 'ui', '--host', '0.0.0.0', '--port', '5000'])
        logger.info("MLflow UI started successfully!")
        logger.info("Press Ctrl+C to stop the UI when you're done viewing results.")
    except Exception as e:
        logger.error(f"Failed to start MLflow UI: {e}")
        logger.info("You can manually start it with: mlflow ui")


def main():
    """Main function with mode selection."""
    parser = argparse.ArgumentParser(description='Quick start script for BERT training')
    
    parser.add_argument('--mode', choices=['quick', 'dev', 'full', 'hp_search', 'custom', 'ui'],
                       default='quick', help='Training mode')
    
    # Custom training options
    parser.add_argument('--model_name', help='Model name for custom training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for custom training')
    parser.add_argument('--batch_size', type=int, help='Batch size for custom training')
    parser.add_argument('--learning_rate', type=float, help='Learning rate for custom training')
    parser.add_argument('--experiment_name', help='MLflow experiment name')
    parser.add_argument('--max_samples', type=int, help='Maximum training samples')
    
    # Utility options
    parser.add_argument('--skip_checks', action='store_true', help='Skip dependency and data checks')
    
    args = parser.parse_args()
    
    # Print welcome message
    print("=" * 60)
    print("🤖 BERT Training on Research Papers - Quick Start")
    print("=" * 60)
    
    # Run checks unless skipped
    if not args.skip_checks:
        logger.info("Checking dependencies and data...")
        
        if not check_dependencies():
            return 1
        
        if not check_data():
            return 1
        
        logger.info("✅ All checks passed!")
    
    # Handle MLflow UI mode
    if args.mode == 'ui':
        start_mlflow_ui()
        return 0
    
    # Run selected training mode
    try:
        if args.mode == 'quick':
            result = run_quick_test()
        elif args.mode == 'dev':
            result = run_development()
        elif args.mode == 'full':
            result = run_full_training()
        elif args.mode == 'hp_search':
            result = run_hyperparameter_search()
        elif args.mode == 'custom':
            result = run_custom_training(args)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
        
        # Check training result
        if result.returncode == 0:
            logger.info("🎉 Training completed successfully!")
            logger.info("To view results, run: python quick_start.py --mode ui")
        else:
            logger.error("❌ Training failed!")
            return result.returncode
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

