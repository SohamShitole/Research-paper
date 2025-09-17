#!/usr/bin/env python3
"""
Model evaluation script for BERT trained on research papers.

This script provides comprehensive evaluation including:
1. Performance metrics on test set
2. Confusion matrix and classification report
3. Sample predictions analysis
4. Error analysis and insights
5. MLflow logging of evaluation results
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch and ML
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertForNextSentencePrediction

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow
import mlflow
import mlflow.pytorch

# Local imports
from bert_trainer import PaperDataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_path: str, task: str = 'next_sentence_prediction'):
        self.model_path = Path(model_path)
        self.task = task
        self.device = device
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        if task == 'next_sentence_prediction':
            self.model = BertForNextSentencePrediction.from_pretrained(str(self.model_path))
        elif task == 'sequence_classification':
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path} for task: {task}")
    
    def evaluate_dataset(self, data_path: str, batch_size: int = 32) -> Dict:
        """Evaluate model on a dataset."""
        # Load dataset
        dataset = PaperDataset(data_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_paper_ids = []
        all_sample_types = []
        
        total_loss = 0.0
        
        logger.info(f"Evaluating on {len(dataset)} samples...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
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
                
                # Get predictions and probabilities
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_paper_ids.extend(batch['paper_id'])
                all_sample_types.extend(batch['sample_type'])
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Additional metrics for binary classification
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'num_samples': len(all_labels)
        }
        
        # ROC AUC for binary classification
        if len(np.unique(all_labels)) == 2:
            probs_positive = [prob[1] for prob in all_probabilities]
            metrics['roc_auc'] = roc_auc_score(all_labels, probs_positive)
        
        # Detailed results
        results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels,
            'paper_ids': all_paper_ids,
            'sample_types': all_sample_types
        }
        
        return results
    
    def generate_confusion_matrix(self, labels: List[int], predictions: List[int], 
                                output_path: str) -> str:
        """Generate and save confusion matrix."""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        return output_path
    
    def generate_classification_report(self, labels: List[int], predictions: List[int]) -> str:
        """Generate detailed classification report."""
        report = classification_report(labels, predictions, output_dict=True)
        
        # Convert to readable format
        report_text = classification_report(labels, predictions)
        
        return report_text, report
    
    def analyze_errors(self, results: Dict) -> Dict:
        """Analyze prediction errors."""
        labels = np.array(results['labels'])
        predictions = np.array(results['predictions'])
        probabilities = np.array(results['probabilities'])
        paper_ids = results['paper_ids']
        sample_types = results['sample_types']
        
        # Find incorrect predictions
        incorrect_mask = labels != predictions
        
        error_analysis = {
            'total_errors': int(np.sum(incorrect_mask)),
            'error_rate': float(np.mean(incorrect_mask)),
            'errors_by_sample_type': {},
            'low_confidence_errors': [],
            'high_confidence_errors': []
        }
        
        # Analyze errors by sample type
        for sample_type in set(sample_types):
            type_mask = np.array(sample_types) == sample_type
            type_errors = np.sum(incorrect_mask & type_mask)
            type_total = np.sum(type_mask)
            
            error_analysis['errors_by_sample_type'][sample_type] = {
                'errors': int(type_errors),
                'total': int(type_total),
                'error_rate': float(type_errors / type_total) if type_total > 0 else 0.0
            }
        
        # Analyze confidence of errors
        for i in range(len(labels)):
            if incorrect_mask[i]:
                max_prob = np.max(probabilities[i])
                error_info = {
                    'paper_id': paper_ids[i],
                    'sample_type': sample_types[i],
                    'true_label': int(labels[i]),
                    'predicted_label': int(predictions[i]),
                    'confidence': float(max_prob),
                    'probabilities': probabilities[i].tolist()
                }
                
                if max_prob < 0.6:  # Low confidence
                    error_analysis['low_confidence_errors'].append(error_info)
                elif max_prob > 0.9:  # High confidence but wrong
                    error_analysis['high_confidence_errors'].append(error_info)
        
        return error_analysis
    
    def generate_roc_curve(self, labels: List[int], probabilities: List[List[float]], 
                          output_path: str) -> str:
        """Generate ROC curve for binary classification."""
        if len(np.unique(labels)) != 2:
            logger.warning("ROC curve only available for binary classification")
            return None
        
        # Extract probabilities for positive class
        probs_positive = [prob[1] for prob in probabilities]
        
        fpr, tpr, _ = roc_curve(labels, probs_positive)
        auc = roc_auc_score(labels, probs_positive)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
        return output_path
    
    def generate_evaluation_report(self, results: Dict, output_dir: str) -> str:
        """Generate comprehensive evaluation report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Basic metrics
        metrics = results['metrics']
        
        # Classification report
        report_text, report_dict = self.generate_classification_report(
            results['labels'], results['predictions']
        )
        
        # Error analysis
        error_analysis = self.analyze_errors(results)
        
        # Generate visualizations
        cm_path = self.generate_confusion_matrix(
            results['labels'], results['predictions'], 
            str(output_dir / 'confusion_matrix.png')
        )
        
        roc_path = None
        if len(np.unique(results['labels'])) == 2:
            roc_path = self.generate_roc_curve(
                results['labels'], results['probabilities'],
                str(output_dir / 'roc_curve.png')
            )
        
        # Create comprehensive report
        report = {
            'model_path': str(self.model_path),
            'task': self.task,
            'evaluation_metrics': metrics,
            'classification_report': report_dict,
            'error_analysis': error_analysis,
            'visualizations': {
                'confusion_matrix': cm_path,
                'roc_curve': roc_path
            }
        }
        
        # Save detailed report
        report_path = output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text report
        text_report_path = output_dir / 'evaluation_report.txt'
        with open(text_report_path, 'w') as f:
            f.write("BERT Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Device: {self.device}\n\n")
            
            f.write("Evaluation Metrics:\n")
            f.write("-" * 20 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n")
            
            f.write("Classification Report:\n")
            f.write("-" * 20 + "\n")
            f.write(report_text)
            f.write("\n")
            
            f.write("Error Analysis:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Errors: {error_analysis['total_errors']}\n")
            f.write(f"Error Rate: {error_analysis['error_rate']:.4f}\n")
            f.write("\nErrors by Sample Type:\n")
            for sample_type, stats in error_analysis['errors_by_sample_type'].items():
                f.write(f"  {sample_type}: {stats['errors']}/{stats['total']} "
                       f"({stats['error_rate']:.4f})\n")
            
            f.write(f"\nLow Confidence Errors: {len(error_analysis['low_confidence_errors'])}\n")
            f.write(f"High Confidence Errors: {len(error_analysis['high_confidence_errors'])}\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
        return str(report_path)
    
    def log_to_mlflow(self, results: Dict, experiment_name: str = "model_evaluation"):
        """Log evaluation results to MLflow."""
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_path", str(self.model_path))
            mlflow.log_param("task", self.task)
            mlflow.log_param("device", str(self.device))
            mlflow.log_param("num_samples", results['metrics']['num_samples'])
            
            # Log metrics
            for key, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_{key}", value)
            
            # Log model
            mlflow.pytorch.log_model(self.model, "evaluated_model")
            
            logger.info("Results logged to MLflow")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate BERT model on research papers')
    
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--test_data', required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--task', default='next_sentence_prediction',
                       choices=['next_sentence_prediction', 'sequence_classification'],
                       help='Model task')
    parser.add_argument('--batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--experiment_name', default='model_evaluation', 
                       help='MLflow experiment name')
    parser.add_argument('--log_mlflow', action='store_true', help='Log results to MLflow')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.task)
    
    # Run evaluation
    logger.info("Starting model evaluation...")
    results = evaluator.evaluate_dataset(args.test_data, args.batch_size)
    
    # Generate report
    report_path = evaluator.generate_evaluation_report(results, args.output_dir)
    
    # Log to MLflow if requested
    if args.log_mlflow:
        evaluator.log_to_mlflow(results, args.experiment_name)
    
    # Print summary
    metrics = results['metrics']
    logger.info("Evaluation completed!")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Loss: {metrics['loss']:.4f}")
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

