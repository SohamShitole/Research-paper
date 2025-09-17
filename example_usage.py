#!/usr/bin/env python3
"""
Example usage scripts for BERT training on research papers.

This file demonstrates various ways to use the training pipeline
with different configurations and use cases.
"""

import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_quick_start():
    """Example: Quick start with minimal configuration."""
    logger.info("Example 1: Quick Start")
    
    # This will run a quick test with 1000 samples and 1 epoch
    cmd = [
        'python3', 'train_bert_papers.py',
        '--quick_test'
    ]
    
    print("Command:", ' '.join(cmd))
    print("This runs a quick test with reduced data for fast experimentation.")
    return cmd


def example_custom_bert():
    """Example: Custom BERT model configuration."""
    logger.info("Example 2: Custom BERT Configuration")
    
    # Train with a different BERT variant
    cmd = [
        'python3', 'train_bert_papers.py',
        '--model_name', 'distilbert-base-uncased',  # Smaller, faster model
        '--epochs', '5',
        '--batch_size', '32',
        '--learning_rate', '3e-5',
        '--experiment_name', 'distilbert_experiment'
    ]
    
    print("Command:", ' '.join(cmd))
    print("This uses DistilBERT (smaller/faster) with custom hyperparameters.")
    return cmd


def example_sequence_classification():
    """Example: Sequence classification task."""
    logger.info("Example 3: Sequence Classification")
    
    # Train for sequence classification instead of NSP
    cmd = [
        'python3', 'train_bert_papers.py',
        '--task', 'sequence_classification',
        '--num_labels', '2',
        '--epochs', '3',
        '--experiment_name', 'sequence_classification_experiment'
    ]
    
    print("Command:", ' '.join(cmd))
    print("This trains BERT for sequence classification instead of next sentence prediction.")
    return cmd


def example_hyperparameter_search():
    """Example: Hyperparameter optimization."""
    logger.info("Example 4: Hyperparameter Search")
    
    # Run hyperparameter search
    cmd = [
        'python3', 'train_bert_papers.py',
        '--run_hyperparameter_search',
        '--max_train_samples', '2000',  # Use smaller dataset for HP search
        '--experiment_name', 'hp_search_experiment'
    ]
    
    print("Command:", ' '.join(cmd))
    print("This runs hyperparameter search to find optimal parameters.")
    return cmd


def example_evaluation():
    """Example: Model evaluation."""
    logger.info("Example 5: Model Evaluation")
    
    # Evaluate a trained model
    cmd = [
        'python3', 'evaluate_model.py',
        '--model_path', 'bert_training_output/models/best_model',
        '--test_data', 'bert_training_output/processed_data/test/tokenized_dataset.json',
        '--output_dir', 'evaluation_results',
        '--log_mlflow'
    ]
    
    print("Command:", ' '.join(cmd))
    print("This evaluates a trained model on the test set with comprehensive metrics.")
    return cmd


def example_preprocessing_only():
    """Example: Data preprocessing only."""
    logger.info("Example 6: Data Preprocessing Only")
    
    # Run only preprocessing
    cmd = [
        'python3', 'data_preprocessing.py'
    ]
    
    print("Command:", ' '.join(cmd))
    print("This runs only the data preprocessing pipeline.")
    return cmd


def example_production_training():
    """Example: Production-ready training configuration."""
    logger.info("Example 7: Production Training")
    
    # Production training with optimal settings
    cmd = [
        'python3', 'train_bert_papers.py',
        '--model_name', 'bert-base-uncased',
        '--epochs', '5',
        '--batch_size', '16',
        '--learning_rate', '2e-5',
        '--warmup_steps', '1000',
        '--weight_decay', '0.01',
        '--gradient_accumulation_steps', '2',
        '--save_every_n_epochs', '1',
        '--experiment_name', 'production_bert_v1',
        '--run_evaluation'
    ]
    
    print("Command:", ' '.join(cmd))
    print("This is a production-ready configuration with optimal hyperparameters.")
    return cmd


def example_mlflow_usage():
    """Example: MLflow experiment tracking."""
    logger.info("Example 8: MLflow Usage")
    
    print("MLflow Commands:")
    print("1. Start MLflow UI: mlflow ui")
    print("2. Set tracking URI: export MLFLOW_TRACKING_URI=http://localhost:5000")
    print("3. View experiments at: http://localhost:5000")
    
    # Example of setting custom MLflow tracking
    cmd = [
        'python3', 'train_bert_papers.py',
        '--experiment_name', 'my_custom_experiment',
        '--mlflow_tracking_uri', 'http://localhost:5000'
    ]
    
    print("Training with custom MLflow setup:", ' '.join(cmd))
    return cmd


def main():
    """Display all examples."""
    print("=" * 80)
    print("🤖 BERT Training Examples - Research Papers")
    print("=" * 80)
    print()
    
    examples = [
        example_quick_start,
        example_custom_bert,
        example_sequence_classification,
        example_hyperparameter_search,
        example_evaluation,
        example_preprocessing_only,
        example_production_training,
        example_mlflow_usage
    ]
    
    for i, example_func in enumerate(examples, 1):
        print(f"\n{'-' * 40}")
        example_func()
        print(f"{'-' * 40}")
        if i < len(examples):
            print()
    
    print("\n" + "=" * 80)
    print("💡 Tips:")
    print("1. Start with quick_test mode to verify everything works")
    print("2. Use MLflow UI to track and compare experiments")
    print("3. Run hyperparameter search before production training")
    print("4. Always evaluate your final model on the test set")
    print("5. Check GPU memory usage and adjust batch_size accordingly")
    print("=" * 80)


if __name__ == "__main__":
    main()

