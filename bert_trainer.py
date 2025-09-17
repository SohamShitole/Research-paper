#!/usr/bin/env python3
"""
BERT Model Training with MLflow Integration.

This module handles:
1. BERT model configuration and initialization
2. Training loop with gradient accumulation
3. MLflow experiment tracking
4. Model checkpointing and evaluation
5. Hyperparameter tuning support
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# PyTorch and ML
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR

# Transformers
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    BertForNextSentencePrediction,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

# MLflow
import mlflow
import mlflow.pytorch
from mlflow import log_metric, log_param, log_artifact

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


class PaperDataset(Dataset):
    """Custom dataset for BERT training on research papers."""
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(item['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long),
            'paper_id': item['paper_id'],
            'sample_type': item['sample_type']
        }


class BERTTrainer:
    """BERT trainer with MLflow integration."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = device
        
        # Initialize MLflow
        mlflow.set_experiment(config.get('experiment_name', 'bert_paper_training'))
        
        # Model setup
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=False)
        
        # Initialize model based on task
        self.task = config.get('task', 'next_sentence_prediction')
        if self.task == 'next_sentence_prediction':
            self.model = BertForNextSentencePrediction.from_pretrained(self.model_name, use_auth_token=False)
        elif self.task == 'sequence_classification':
            num_labels = config.get('num_labels', 2)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=num_labels,
                use_auth_token=False
            )
        else:
            raise ValueError(f"Unsupported task: {self.task}")
        
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = config.get('epochs', 3)
        self.batch_size = config.get('batch_size', 16)
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.warmup_steps = config.get('warmup_steps', 500)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        
        # Paths
        self.output_dir = Path(config.get('output_dir', 'bert_models'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.best_eval_accuracy = 0.0
        
        # Metrics tracking
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
    
    def create_data_loaders(self, train_data_path: str, val_data_path: Optional[str] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create data loaders for training and validation."""
        
        # Load training data
        train_dataset = PaperDataset(train_data_path, self.config.get('max_train_samples'))
        
        # Split training data if no validation data provided
        if val_data_path is None:
            train_size = int(0.9 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        else:
            val_dataset = PaperDataset(val_data_path, self.config.get('max_val_samples'))
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        ) if val_dataset else None
        
        logger.info(f"Created train loader with {len(train_loader)} batches")
        if val_loader:
            logger.info(f"Created validation loader with {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def setup_optimizer_and_scheduler(self, train_loader: DataLoader):
        """Setup optimizer and learning rate scheduler."""
        
        # Calculate total steps
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8
        )
        
        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Setup optimizer and scheduler for {total_steps} steps")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss / self.gradient_accumulation_steps
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log to MLflow
                if self.global_step % 100 == 0:
                    log_metric("train_loss", loss.item() * self.gradient_accumulation_steps, step=self.global_step)
                    log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1
        }
        
        self.eval_losses.append(avg_loss)
        self.eval_accuracies.append(accuracy)
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / f'checkpoint-epoch-{epoch+1}'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
            'best_eval_accuracy': self.best_eval_accuracy,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'eval_accuracies': self.eval_accuracies,
            'config': self.config
        }, checkpoint_dir / 'training_state.pt')
        
        # Log checkpoint to MLflow
        mlflow.log_artifacts(str(checkpoint_dir), f'checkpoint-epoch-{epoch+1}')
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if self.eval_losses:
            ax1.plot(epochs, self.eval_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curve
        if self.eval_accuracies:
            ax2.plot(epochs, self.eval_accuracies, 'g-', label='Validation Accuracy')
            ax2.set_title('Validation Accuracy')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to MLflow
        log_artifact(str(plot_path))
        
        logger.info(f"Saved training curves to {plot_path}")
    
    def train(self, train_data_path: str, val_data_path: Optional[str] = None):
        """Main training loop."""
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            for key, value in self.config.items():
                log_param(key, value)
            
            log_param("device", str(self.device))
            log_param("model_name", self.model_name)
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(train_data_path, val_data_path)
            
            # Setup optimizer and scheduler
            self.setup_optimizer_and_scheduler(train_loader)
            
            # Training loop
            logger.info("Starting training...")
            start_time = time.time()
            
            for epoch in range(self.epochs):
                # Train
                train_loss = self.train_epoch(train_loader, epoch)
                
                # Evaluate
                if val_loader:
                    eval_metrics = self.evaluate(val_loader)
                    
                    # Log metrics
                    log_metric("epoch_train_loss", train_loss, step=epoch)
                    for metric_name, metric_value in eval_metrics.items():
                        log_metric(metric_name, metric_value, step=epoch)
                    
                    # Check for best model
                    if eval_metrics['eval_accuracy'] > self.best_eval_accuracy:
                        self.best_eval_accuracy = eval_metrics['eval_accuracy']
                        self.best_eval_loss = eval_metrics['eval_loss']
                        
                        # Save best model
                        best_model_dir = self.output_dir / 'best_model'
                        best_model_dir.mkdir(exist_ok=True)
                        self.model.save_pretrained(best_model_dir)
                        self.tokenizer.save_pretrained(best_model_dir)
                        
                        # Log best model to MLflow
                        mlflow.pytorch.log_model(self.model, "best_model")
                    
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                              f"Train Loss: {train_loss:.4f}, "
                              f"Val Loss: {eval_metrics['eval_loss']:.4f}, "
                              f"Val Acc: {eval_metrics['eval_accuracy']:.4f}")
                else:
                    log_metric("epoch_train_loss", train_loss, step=epoch)
                    logger.info(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.get('save_every_n_epochs', 1) == 0:
                    metrics = eval_metrics if val_loader else {'train_loss': train_loss}
                    self.save_checkpoint(epoch, metrics)
            
            # Training completed
            training_time = time.time() - start_time
            log_metric("training_time_seconds", training_time)
            log_metric("best_eval_accuracy", self.best_eval_accuracy)
            log_metric("best_eval_loss", self.best_eval_loss)
            
            # Plot training curves
            self.plot_training_curves()
            
            # Save final model
            final_model_dir = self.output_dir / 'final_model'
            final_model_dir.mkdir(exist_ok=True)
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            
            # Log final model to MLflow
            mlflow.pytorch.log_model(self.model, "final_model")
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            logger.info(f"Best validation accuracy: {self.best_eval_accuracy:.4f}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train BERT on research papers')
    
    # Data arguments
    parser.add_argument('--train_data', required=True, help='Path to training data')
    parser.add_argument('--val_data', help='Path to validation data')
    parser.add_argument('--output_dir', default='bert_models', help='Output directory')
    
    # Model arguments
    parser.add_argument('--model_name', default='bert-base-uncased', help='Pretrained model name')
    parser.add_argument('--task', default='next_sentence_prediction', 
                       choices=['next_sentence_prediction', 'sequence_classification'],
                       help='Training task')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels for classification')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    parser.add_argument('--save_every_n_epochs', type=int, default=1, help='Save checkpoint every N epochs')
    
    # Data limits
    parser.add_argument('--max_train_samples', type=int, help='Maximum training samples')
    parser.add_argument('--max_val_samples', type=int, help='Maximum validation samples')
    
    # MLflow
    parser.add_argument('--experiment_name', default='bert_paper_training', help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Initialize trainer
    trainer = BERTTrainer(config)
    
    # Start training
    trainer.train(args.train_data, args.val_data)


if __name__ == "__main__":
    main()

