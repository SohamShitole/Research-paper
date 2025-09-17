# NVIDIA NIM Integration for Research Paper Analysis

This integration adds NVIDIA NIM (NVIDIA Inference Microservices) support to your research paper analysis pipeline, enabling you to use state-of-the-art open source language models for enhanced paper analysis.

## 🚀 Quick Start

The fastest way to get started is with the quick start script:

```bash
python quick_start_nim.py
```

This interactive script will:
- Check requirements and install missing dependencies
- Help you choose the best model for your use case
- Deploy the NIM service
- Run sample analysis
- Analyze papers from your existing pipeline

## 📋 Prerequisites

### System Requirements
- **NVIDIA GPU**: RTX 3090, RTX 4090, A100, or similar (8GB+ VRAM)
- **Docker**: Latest version with NVIDIA Container Toolkit
- **Python**: 3.8 or higher
- **Memory**: 16GB+ RAM recommended
- **Storage**: 20GB+ free space for model caching

### Software Setup

1. **Install Docker and NVIDIA Container Toolkit**:
   ```bash
   # Install Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install NVIDIA Container Toolkit
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
     sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Get NVIDIA NGC API Key**:
   - Visit [NVIDIA NGC](https://ngc.nvidia.com/setup/api-key)
   - Create account and generate API key
   - Set environment variable:
   ```bash
   export NGC_API_KEY=your_api_key_here
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🤖 Available Models

| Model | Size | Memory Req | Best For | Speed |
|-------|------|------------|----------|-------|
| **llama-3.1-8b-instruct** | 8B | 16GB | General analysis, balanced performance | Medium |
| **codellama-13b-instruct** | 13B | 24GB | Technical papers, code analysis | Slower |
| **mistral-7b-instruct** | 7B | 14GB | Fast batch processing | Fastest |

## 🛠️ Manual Setup

### 1. Deploy NIM Service

```bash
# Check prerequisites
python deploy_nim_service.py check

# List available models
python deploy_nim_service.py list

# Deploy a specific model
python deploy_nim_service.py deploy --model llama-3.1-8b-instruct

# Check service status
python deploy_nim_service.py status
```

### 2. Test Connection

```bash
# Test basic functionality
python nvidia_nim_analyzer.py

# Or check service directly
python deploy_nim_service.py check
```

## 📊 Usage Examples

### Basic Analysis

```python
from nvidia_nim_analyzer import NVIDIANIMAnalyzer, NIMConfig

# Initialize analyzer
config = NIMConfig(model_name="llama-3.1-8b-instruct")
analyzer = NVIDIANIMAnalyzer(config)

# Analyze a paper
result = analyzer.analyze_paper_text(
    paper_text="Your research paper content here...",
    paper_id="paper_001",
    title="Example Paper Title"
)

print(f"Summary: {result.summary}")
print(f"Key Findings: {result.key_findings}")
print(f"Confidence: {result.confidence_score}")
```

### Batch Analysis

```python
# Analyze multiple papers
papers = [
    {"text": "Paper 1 content...", "id": "p1", "title": "Title 1"},
    {"text": "Paper 2 content...", "id": "p2", "title": "Title 2"},
]

results = analyzer.analyze_paper_batch(papers, max_concurrent=3)
```

### Integration with Existing Pipeline

```bash
# Analyze papers from your existing pipeline
python nim_paper_pipeline.py \
    --data-dir bert_training_output/processed_data \
    --model llama-3.1-8b-instruct \
    --max-papers 50 \
    --batch-size 5
```

## 📁 File Structure

```
archive-2/
├── nvidia_nim_analyzer.py          # Core NIM analysis functionality
├── deploy_nim_service.py           # NIM service deployment manager
├── nim_paper_pipeline.py           # Integration with existing pipeline
├── quick_start_nim.py              # Interactive setup script
├── nim_analysis_output/            # Analysis results (created during use)
│   ├── nim_analysis_results_*.json # Detailed analysis results
│   ├── nim_analysis_report_*.json  # Comprehensive reports
│   └── nim_analysis_summary_*.csv  # Summary data for spreadsheets
└── README_NIM_INTEGRATION.md       # This file
```

## 🔧 Configuration Options

### NIMConfig Parameters

```python
config = NIMConfig(
    base_url="http://localhost:8000/v1",  # NIM service URL
    model_name="llama-3.1-8b-instruct",  # Model to use
    max_tokens=2048,                      # Max response tokens
    temperature=0.7,                      # Response creativity (0-1)
    timeout=60                            # Request timeout (seconds)
)
```

### Pipeline Options

```bash
# Common pipeline options
python nim_paper_pipeline.py \
    --model llama-3.1-8b-instruct \     # Choose model
    --max-papers 100 \                   # Limit papers analyzed
    --batch-size 3 \                     # Concurrent analyses
    --output-dir custom_output \         # Custom output directory
    --nim-url http://localhost:8000/v1   # Custom NIM URL
```

## 📊 Output Analysis

### JSON Results Structure

```json
{
  "paper_id": "unique_paper_id",
  "title": "Paper Title",
  "summary": "AI-generated summary",
  "key_findings": ["finding 1", "finding 2"],
  "methodology": "Research methodology description",
  "limitations": ["limitation 1", "limitation 2"],
  "future_work": ["direction 1", "direction 2"],
  "sentiment_score": 0.8,
  "confidence_score": 0.9,
  "processing_time": 2.5,
  "model_used": "llama-3.1-8b-instruct"
}
```

### Comprehensive Report

Each analysis generates:
- **Detailed Results**: Individual paper analyses
- **Summary Report**: Aggregate statistics and insights
- **CSV Export**: Spreadsheet-friendly summary data
- **BERT Comparison**: Performance comparison with existing BERT results

## 🎯 Performance Optimization

### Model Selection Guidelines

- **For Technical Papers**: Use `codellama-13b-instruct`
- **For General Research**: Use `llama-3.1-8b-instruct`
- **For Batch Processing**: Use `mistral-7b-instruct`

### Batch Processing Tips

```bash
# Optimal batch processing
python nim_paper_pipeline.py \
    --model mistral-7b-instruct \
    --batch-size 5 \
    --max-papers 1000
```

### Memory Management

- Monitor GPU memory usage: `nvidia-smi`
- Adjust batch size based on available memory
- Use smaller models for large datasets

## 🔍 Troubleshooting

### Common Issues

1. **"Cannot connect to NIM service"**
   ```bash
   # Check if service is running
   docker ps | grep nim
   
   # Check service logs
   docker logs nim-llama-3-1-8b-instruct
   
   # Restart service
   python deploy_nim_service.py deploy --model llama-3.1-8b-instruct
   ```

2. **"CUDA out of memory"**
   ```bash
   # Use smaller model
   python deploy_nim_service.py deploy --model mistral-7b-instruct
   
   # Or reduce batch size
   python nim_paper_pipeline.py --batch-size 1
   ```

3. **"NGC API key not set"**
   ```bash
   export NGC_API_KEY=your_key_here
   ```

4. **"Docker not found"**
   - Install Docker: https://docs.docker.com/get-docker/
   - Install NVIDIA Container Toolkit

### Service Management

```bash
# Check all running containers
python deploy_nim_service.py status

# Stop specific model
python deploy_nim_service.py stop --model llama-3.1-8b-instruct

# Clean up all NIM containers
python deploy_nim_service.py cleanup
```

## 📈 Comparison with BERT

| Feature | BERT Pipeline | NIM Integration |
|---------|---------------|-----------------|
| **Setup Time** | Hours (training) | Minutes (deployment) |
| **Model Size** | 440MB | 8-13GB |
| **Analysis Depth** | Task-specific | Comprehensive |
| **Flexibility** | Fixed tasks | Natural language queries |
| **Hardware Req** | 8GB GPU | 16-24GB GPU |
| **Inference Speed** | Very Fast | Fast |
| **Interpretability** | Limited | High |

## 🚀 Advanced Usage

### Custom Analysis Prompts

```python
# Modify system prompt for specific analysis types
analyzer._get_system_prompt = lambda: """
You are a specialist in machine learning papers. 
Focus on methodology, experimental design, and reproducibility.
"""
```

### Integration with MLflow

```python
import mlflow

# Log NIM results to MLflow
with mlflow.start_run():
    results = analyzer.analyze_paper_batch(papers)
    mlflow.log_metric("papers_analyzed", len(results))
    mlflow.log_metric("avg_confidence", sum(r.confidence_score for r in results) / len(results))
```

### Custom Model Deployment

```bash
# Deploy with custom settings
docker run --rm --gpus all \
    --shm-size=24GB \
    --network=host \
    -e NGC_API_KEY=$NGC_API_KEY \
    -v ~/nim_cache:/opt/nim/.cache \
    nvcr.io/nim/meta/llama-3.1-8b-instruct:1.0.0
```

## 📚 Additional Resources

- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NGC Catalog](https://catalog.ngc.nvidia.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Docker NVIDIA Runtime](https://github.com/NVIDIA/nvidia-container-runtime)

## 🤝 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Review service logs: `docker logs <container_name>`
3. Test with sample analysis first
4. Verify GPU and Docker setup

---

**Happy Analyzing! 🎉**
