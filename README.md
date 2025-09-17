# BERT Training on Research Papers

This repository provides a complete pipeline for training BERT models on research papers from the SciTLDR dataset with comprehensive MLflow experiment tracking.

## 🚀 Features

- **Complete Data Pipeline**: PDF text extraction, cleaning, and preprocessing
- **BERT Training**: Next Sentence Prediction and Sequence Classification tasks
- **MLflow Integration**: Comprehensive experiment tracking and model management
- **Evaluation Suite**: Detailed model evaluation with metrics and visualizations
- **Hyperparameter Optimization**: Grid search and parameter tuning support
- **Scalable Architecture**: Supports large datasets with efficient data loading

## 📁 Project Structure

```
archive-2/
├── download_papers.py          # Paper downloading script
├── data_preprocessing.py       # PDF text extraction and preprocessing
├── bert_trainer.py            # BERT model training with MLflow
├── train_bert_papers.py       # Main training pipeline
├── evaluate_model.py          # Model evaluation and analysis
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── downloaded_papers/         # Downloaded papers directory
│   ├── train/                # Training papers
│   ├── val/                  # Validation papers
│   └── test/                 # Test papers
└── bert_training_output/      # Training outputs (created during training)
    ├── processed_data/       # Preprocessed datasets
    ├── models/              # Trained models
    └── logs/                # Training logs
```

## 🛠️ Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd archive-2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download NLTK data** (will be done automatically on first run):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## 📊 Data Overview

The repository contains **1,982 training papers**, **618 validation papers**, and **618 test papers** downloaded from the SciTLDR dataset. Each paper includes:
- PDF content
- Metadata (title, authors, abstract, venue, year)
- OpenReview links where available

## 🎯 Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the complete pipeline with default settings:

```bash
python train_bert_papers.py
```

### Option 2: Quick Test Mode

For testing with reduced data and epochs:

```bash
python train_bert_papers.py --quick_test
```

### Option 3: Custom Configuration

```bash
python train_bert_papers.py \
    --model_name bert-base-uncased \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --experiment_name my_bert_experiment
```

## 📈 MLflow Tracking

The pipeline automatically tracks all experiments with MLflow. After training, you can view results:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to see:
- Training metrics and loss curves
- Hyperparameters for each run
- Model artifacts and checkpoints
- Evaluation results and visualizations

## 🔧 Advanced Usage

### 1. Data Preprocessing Only

```bash
python data_preprocessing.py
```

### 2. Training with Custom Data

```bash
python bert_trainer.py \
    --train_data processed_data/tokenized_dataset.json \
    --val_data processed_data/val_tokenized_dataset.json \
    --output_dir my_models
```

### 3. Model Evaluation

```bash
python evaluate_model.py \
    --model_path bert_training_output/models/best_model \
    --test_data processed_data/test_tokenized_dataset.json \
    --output_dir evaluation_results
```

### 4. Hyperparameter Search

```bash
python train_bert_papers.py --run_hyperparameter_search
```

## 🎛️ Configuration Options

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch_size` | 16 | Training batch size |
| `--learning_rate` | 2e-5 | Learning rate |
| `--warmup_steps` | 500 | Warmup steps for learning rate scheduler |
| `--weight_decay` | 0.01 | Weight decay for optimization |
| `--max_length` | 512 | Maximum sequence length |

### Model Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | bert-base-uncased | Pretrained model name |
| `--task` | next_sentence_prediction | Training task |
| `--num_labels` | 2 | Number of labels (for classification) |

### Data Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_train_samples` | None | Limit training samples (for testing) |
| `--max_val_samples` | None | Limit validation samples |
| `--skip_preprocessing` | False | Skip data preprocessing |

## 📊 Training Tasks

### 1. Next Sentence Prediction (NSP)

The default task trains BERT to predict whether two sentences are consecutive in the original paper. This helps the model learn document structure and coherence.

**Sample Types Created:**
- Abstract + Introduction pairs
- Title + Abstract pairs  
- Consecutive sentence pairs
- Random sentence pairs (negative examples)

### 2. Sequence Classification

Alternative task for document classification or similarity tasks:

```bash
python train_bert_papers.py --task sequence_classification --num_labels 2
```

## 📈 Monitoring and Logging

### MLflow Features

- **Experiment Tracking**: All runs automatically logged
- **Parameter Logging**: Hyperparameters, model config, data info
- **Metric Tracking**: Loss, accuracy, F1, precision, recall
- **Model Artifacts**: Best models, checkpoints, tokenizers
- **Visualizations**: Training curves, confusion matrices, ROC curves

### Log Files

- `training.log`: Detailed training logs
- `pipeline_summary.json`: Complete pipeline summary
- `evaluation_report.json`: Detailed evaluation results

## 🎯 Model Performance

### Expected Results

With default settings on the research papers dataset:

- **Training Time**: ~2-4 hours (depending on hardware)
- **Memory Usage**: ~8-16GB GPU memory
- **Validation Accuracy**: ~85-92% (varies by task and data)
- **Model Size**: ~440MB (BERT-base)

### Performance Tips

1. **GPU Acceleration**: Use CUDA-enabled GPU for faster training
2. **Batch Size**: Increase batch size if you have more GPU memory
3. **Gradient Accumulation**: Use gradient accumulation for larger effective batch sizes
4. **Mixed Precision**: Enable mixed precision training for speed (can be added)

## 🔍 Evaluation and Analysis

The evaluation script provides comprehensive analysis:

```bash
python evaluate_model.py --model_path bert_training_output/models/best_model --test_data processed_data/test/tokenized_dataset.json
```

**Evaluation Outputs:**
- Accuracy, Precision, Recall, F1-score
- Confusion matrix visualization
- ROC curve (for binary classification)
- Error analysis by sample type
- Classification report
- Low/high confidence error analysis

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   ```bash
   python train_bert_papers.py --batch_size 8 --gradient_accumulation_steps 2
   ```

2. **PDF Processing Errors**:
   - Some PDFs may fail to extract text
   - Check `processed_data/` logs for failed extractions
   - Failed papers are automatically skipped

3. **MLflow UI Not Starting**:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

4. **Memory Issues During Preprocessing**:
   ```bash
   python train_bert_papers.py --max_train_samples 1000
   ```

### System Requirements

- **Python**: 3.8+
- **Memory**: 16GB+ RAM recommended
- **GPU**: 8GB+ VRAM recommended (can run on CPU)
- **Storage**: 10GB+ free space

## 📚 Example Workflows

### Research Experiment Workflow

```bash
# 1. Full pipeline with hyperparameter search
python train_bert_papers.py \
    --run_hyperparameter_search \
    --experiment_name research_experiment_v1

# 2. Train final model with best parameters
python train_bert_papers.py \
    --learning_rate 3e-5 \
    --batch_size 32 \
    --epochs 5 \
    --experiment_name final_model_v1

# 3. Evaluate on test set
python evaluate_model.py \
    --model_path bert_training_output/models/best_model \
    --test_data processed_data/test/tokenized_dataset.json \
    --log_mlflow
```

### Quick Development Workflow

```bash
# Quick test with small dataset
python train_bert_papers.py --quick_test --experiment_name dev_test

# Check results
mlflow ui
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SciTLDR Dataset**: Research paper dataset
- **Hugging Face Transformers**: BERT implementation
- **MLflow**: Experiment tracking
- **OpenReview**: Paper metadata and PDFs

## 📧 Support

For questions or issues:
1. Check the troubleshooting section
2. Review MLflow logs for training issues
3. Check GitHub issues for known problems
4. Create a new issue with detailed information

---

**Happy Training! 🚀**# Research-paper
