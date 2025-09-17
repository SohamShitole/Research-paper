#!/usr/bin/env python3
"""
Main training script for BERT on research papers.

This script orchestrates the complete pipeline:
1. Data preprocessing and text extraction
2. Dataset creation and tokenization
3. BERT model training with MLflow tracking
4. Model evaluation and metrics logging
5. Hyperparameter optimization support
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# MLflow
import mlflow
import mlflow.pytorch

# Local imports
from data_preprocessing import PaperDatasetCreator, BERTDatasetTokenizer
from bert_trainer import BERTTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PaperBERTTrainingPipeline:
    """Complete pipeline for training BERT on research papers."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_dir = Path(config.get('base_dir', 'downloaded_papers'))
        self.output_dir = Path(config.get('output_dir', 'bert_training_output'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'processed_data').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # Set MLflow tracking URI if specified
        if config.get('mlflow_tracking_uri'):
            mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
        
        logger.info(f"Initialized pipeline with output directory: {self.output_dir}")
    
    def preprocess_data(self) -> Dict[str, str]:
        """Run data preprocessing pipeline."""
        logger.info("Starting data preprocessing...")
        
        processed_data_dir = self.output_dir / 'processed_data'
        
        # Initialize dataset creator
        creator = PaperDatasetCreator(str(self.base_dir))
        
        data_paths = {}
        
        # Process training data
        if (self.base_dir / 'train').exists():
            logger.info("Processing training papers...")
            train_papers = creator.process_papers('train')
            
            if train_papers:
                # Save processed papers
                train_papers_path = processed_data_dir / 'train_papers.json'
                creator.save_dataset(train_papers, str(train_papers_path))
                
                # Create BERT dataset
                bert_train_dataset = creator.create_bert_dataset(
                    train_papers, 
                    max_length=self.config.get('max_length', 512)
                )
                
                bert_train_path = processed_data_dir / 'bert_train_dataset.json'
                creator.save_bert_dataset(bert_train_dataset, str(bert_train_path))
                
                # Tokenize dataset
                tokenizer = BERTDatasetTokenizer(self.config.get('model_name', 'bert-base-uncased'))
                tokenized_train_path = tokenizer.tokenize_dataset(
                    str(bert_train_path),
                    str(processed_data_dir / 'train'),
                    max_length=self.config.get('max_length', 512)
                )
                
                data_paths['train'] = tokenized_train_path
                logger.info(f"Training data processed: {len(train_papers)} papers, {len(bert_train_dataset)} samples")
        
        # Process validation data
        if (self.base_dir / 'val').exists():
            logger.info("Processing validation papers...")
            val_papers = creator.process_papers('val')
            
            if val_papers:
                # Save processed papers
                val_papers_path = processed_data_dir / 'val_papers.json'
                creator.save_dataset(val_papers, str(val_papers_path))
                
                # Create BERT dataset
                bert_val_dataset = creator.create_bert_dataset(
                    val_papers,
                    max_length=self.config.get('max_length', 512)
                )
                
                bert_val_path = processed_data_dir / 'bert_val_dataset.json'
                creator.save_bert_dataset(bert_val_dataset, str(bert_val_path))
                
                # Tokenize dataset
                tokenizer = BERTDatasetTokenizer(self.config.get('model_name', 'bert-base-uncased'))
                tokenized_val_path = tokenizer.tokenize_dataset(
                    str(bert_val_path),
                    str(processed_data_dir / 'val'),
                    max_length=self.config.get('max_length', 512)
                )
                
                data_paths['val'] = tokenized_val_path
                logger.info(f"Validation data processed: {len(val_papers)} papers, {len(bert_val_dataset)} samples")
        
        # Process test data
        if (self.base_dir / 'test').exists():
            logger.info("Processing test papers...")
            test_papers = creator.process_papers('test')
            
            if test_papers:
                # Save processed papers
                test_papers_path = processed_data_dir / 'test_papers.json'
                creator.save_dataset(test_papers, str(test_papers_path))
                
                # Create BERT dataset
                bert_test_dataset = creator.create_bert_dataset(
                    test_papers,
                    max_length=self.config.get('max_length', 512)
                )
                
                bert_test_path = processed_data_dir / 'bert_test_dataset.json'
                creator.save_bert_dataset(bert_test_dataset, str(bert_test_path))
                
                # Tokenize dataset
                tokenizer = BERTDatasetTokenizer(self.config.get('model_name', 'bert-base-uncased'))
                tokenized_test_path = tokenizer.tokenize_dataset(
                    str(bert_test_path),
                    str(processed_data_dir / 'test'),
                    max_length=self.config.get('max_length', 512)
                )
                
                data_paths['test'] = tokenized_test_path
                logger.info(f"Test data processed: {len(test_papers)} papers, {len(bert_test_dataset)} samples")
        
        # Save data paths
        with open(self.output_dir / 'data_paths.json', 'w') as f:
            json.dump(data_paths, f, indent=2)
        
        logger.info("Data preprocessing completed!")
        return data_paths
    
    def train_model(self, data_paths: Dict[str, str]) -> str:
        """Train BERT model."""
        logger.info("Starting model training...")
        
        # Prepare training config
        training_config = {
            'model_name': self.config.get('model_name', 'bert-base-uncased'),
            'task': self.config.get('task', 'next_sentence_prediction'),
            'num_labels': self.config.get('num_labels', 2),
            'epochs': self.config.get('epochs', 3),
            'batch_size': self.config.get('batch_size', 16),
            'learning_rate': self.config.get('learning_rate', 2e-5),
            'warmup_steps': self.config.get('warmup_steps', 500),
            'weight_decay': self.config.get('weight_decay', 0.01),
            'gradient_accumulation_steps': self.config.get('gradient_accumulation_steps', 1),
            'max_grad_norm': self.config.get('max_grad_norm', 1.0),
            'save_every_n_epochs': self.config.get('save_every_n_epochs', 1),
            'max_train_samples': self.config.get('max_train_samples'),
            'max_val_samples': self.config.get('max_val_samples'),
            'experiment_name': self.config.get('experiment_name', 'bert_paper_training'),
            'output_dir': str(self.output_dir / 'models')
        }
        
        # Initialize trainer
        trainer = BERTTrainer(training_config)
        
        # Start training
        trainer.train(
            train_data_path=data_paths['train'],
            val_data_path=data_paths.get('val')
        )
        
        best_model_path = self.output_dir / 'models' / 'best_model'
        logger.info(f"Model training completed! Best model saved to: {best_model_path}")
        
        return str(best_model_path)
    
    def evaluate_model(self, model_path: str, data_paths: Dict[str, str]):
        """Evaluate trained model on test set."""
        if 'test' not in data_paths:
            logger.warning("No test data available for evaluation")
            return
        
        logger.info("Starting model evaluation...")
        
        # TODO: Implement detailed evaluation
        # This would include:
        # - Loading the trained model
        # - Running inference on test set
        # - Computing detailed metrics
        # - Generating evaluation report
        # - Logging results to MLflow
        
        logger.info("Model evaluation completed!")
    
    def run_hyperparameter_search(self, data_paths: Dict[str, str]):
        """Run hyperparameter optimization."""
        logger.info("Starting hyperparameter search...")
        
        # Define hyperparameter search space
        search_space = {
            'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
            'batch_size': [8, 16, 32],
            'warmup_steps': [100, 500, 1000],
            'weight_decay': [0.0, 0.01, 0.1],
        }
        
        best_score = 0.0
        best_params = {}
        
        # Simple grid search (could be replaced with more sophisticated methods)
        for lr in search_space['learning_rate']:
            for bs in search_space['batch_size']:
                for ws in search_space['warmup_steps']:
                    for wd in search_space['weight_decay']:
                        
                        # Create config for this run
                        run_config = self.config.copy()
                        run_config.update({
                            'learning_rate': lr,
                            'batch_size': bs,
                            'warmup_steps': ws,
                            'weight_decay': wd,
                            'epochs': 1,  # Quick runs for hyperparameter search
                            'experiment_name': f"{self.config.get('experiment_name', 'bert_paper_training')}_hp_search"
                        })
                        
                        logger.info(f"Testing hyperparameters: lr={lr}, bs={bs}, ws={ws}, wd={wd}")
                        
                        # Train with these hyperparameters
                        trainer = BERTTrainer(run_config)
                        trainer.train(
                            train_data_path=data_paths['train'],
                            val_data_path=data_paths.get('val')
                        )
                        
                        # Get validation score
                        if trainer.best_eval_accuracy > best_score:
                            best_score = trainer.best_eval_accuracy
                            best_params = {
                                'learning_rate': lr,
                                'batch_size': bs,
                                'warmup_steps': ws,
                                'weight_decay': wd
                            }
        
        logger.info(f"Best hyperparameters found: {best_params}")
        logger.info(f"Best validation accuracy: {best_score:.4f}")
        
        # Save best hyperparameters
        with open(self.output_dir / 'best_hyperparameters.json', 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_score': best_score
            }, f, indent=2)
        
        return best_params
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline."""
        logger.info("Starting complete BERT training pipeline...")
        start_time = time.time()
        
        try:
            # Step 1: Preprocess data
            if self.config.get('skip_preprocessing', False):
                # Load existing data paths
                data_paths_file = self.output_dir / 'data_paths.json'
                if data_paths_file.exists():
                    with open(data_paths_file, 'r') as f:
                        data_paths = json.load(f)
                    logger.info("Skipped preprocessing, loaded existing data paths")
                else:
                    logger.error("Skip preprocessing requested but no existing data paths found")
                    return
            else:
                data_paths = self.preprocess_data()
            
            if not data_paths.get('train'):
                logger.error("No training data available!")
                return
            
            # Step 2: Hyperparameter search (optional)
            if self.config.get('run_hyperparameter_search', False):
                best_params = self.run_hyperparameter_search(data_paths)
                # Update config with best parameters
                self.config.update(best_params)
            
            # Step 3: Train model
            model_path = self.train_model(data_paths)
            
            # Step 4: Evaluate model (optional)
            if self.config.get('run_evaluation', True):
                self.evaluate_model(model_path, data_paths)
            
            # Pipeline completed
            total_time = time.time() - start_time
            logger.info(f"Complete pipeline finished in {total_time:.2f} seconds")
            
            # Save pipeline summary
            summary = {
                'pipeline_config': self.config,
                'data_paths': data_paths,
                'model_path': model_path,
                'total_time_seconds': total_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(self.output_dir / 'pipeline_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Pipeline summary saved to: {self.output_dir / 'pipeline_summary.json'}")
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            raise


def main():
    """Main function with comprehensive argument parsing."""
    parser = argparse.ArgumentParser(description='Train BERT on research papers - Complete Pipeline')
    
    # Data arguments
    parser.add_argument('--base_dir', default='downloaded_papers', 
                       help='Base directory with downloaded papers')
    parser.add_argument('--output_dir', default='bert_training_output', 
                       help='Output directory for all results')
    
    # Preprocessing arguments
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip preprocessing and use existing processed data')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for tokenization')
    
    # Model arguments
    parser.add_argument('--model_name', default='bert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--task', default='next_sentence_prediction',
                       choices=['next_sentence_prediction', 'sequence_classification'],
                       help='Training task')
    parser.add_argument('--num_labels', type=int, default=2,
                       help='Number of labels for classification')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--save_every_n_epochs', type=int, default=1,
                       help='Save checkpoint every N epochs')
    
    # Data limits
    parser.add_argument('--max_train_samples', type=int,
                       help='Maximum training samples (for quick testing)')
    parser.add_argument('--max_val_samples', type=int,
                       help='Maximum validation samples')
    
    # Pipeline control
    parser.add_argument('--run_hyperparameter_search', action='store_true',
                       help='Run hyperparameter search before final training')
    parser.add_argument('--run_evaluation', action='store_true', default=True,
                       help='Run evaluation on test set')
    
    # MLflow arguments
    parser.add_argument('--experiment_name', default='bert_paper_training',
                       help='MLflow experiment name')
    parser.add_argument('--mlflow_tracking_uri', 
                       help='MLflow tracking URI (default: local)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode with reduced data and epochs')
    
    args = parser.parse_args()
    
    # Quick test mode adjustments
    if args.quick_test:
        args.epochs = 1
        args.max_train_samples = 1000
        args.max_val_samples = 200
        args.batch_size = 8
        logger.info("Running in quick test mode")
    
    # Convert args to config dict
    config = vars(args)
    
    # Initialize and run pipeline
    pipeline = PaperBERTTrainingPipeline(config)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()

