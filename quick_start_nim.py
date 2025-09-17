#!/usr/bin/env python3
"""
Quick Start Script for NVIDIA NIM Research Paper Analysis

This script provides a simple way to get started with NVIDIA NIM
for research paper analysis with minimal configuration.

Author: Generated for research paper analysis pipeline
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_header():
    """Print welcome header."""
    print("🚀 NVIDIA NIM Quick Start for Research Paper Analysis")
    print("=" * 60)
    print("This script will help you get started with NVIDIA NIM for")
    print("analyzing research papers in your existing pipeline.")
    print("=" * 60)

def check_requirements():
    """Check if basic requirements are met."""
    print("\n🔍 Checking requirements...")
    
    # Check Python modules
    required_modules = ['openai', 'pandas', 'requests']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} - installed")
        except ImportError:
            print(f"❌ {module} - missing")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n📦 Installing missing modules: {', '.join(missing_modules)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_modules)
            print("✅ Modules installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install modules. Please install manually:")
            print(f"pip install {' '.join(missing_modules)}")
            return False
    
    return True

def setup_environment():
    """Setup environment variables and directories."""
    print("\n🛠️  Setting up environment...")
    
    # Check NGC API key
    ngc_key = os.getenv('NGC_API_KEY')
    if not ngc_key:
        print("⚠️  NGC_API_KEY not found in environment")
        print("You'll need an NVIDIA NGC API key to use NIM services.")
        print("Get one from: https://ngc.nvidia.com/setup/api-key")
        
        key_input = input("Enter your NGC API key (or press Enter to skip): ").strip()
        if key_input:
            os.environ['NGC_API_KEY'] = key_input
            print("✅ NGC API key set for this session")
        else:
            print("⚠️  Skipping NGC API key setup")
    else:
        print("✅ NGC_API_KEY found in environment")
    
    # Setup cache directory
    cache_path = os.path.expanduser("~/nim_cache")
    Path(cache_path).mkdir(parents=True, exist_ok=True)
    os.environ['NIM_CACHE_PATH'] = cache_path
    print(f"✅ NIM cache directory: {cache_path}")
    
    return True

def choose_model():
    """Let user choose which model to use."""
    print("\n🤖 Choose a NIM model:")
    print("1. llama-3.1-8b-instruct (Recommended - balanced performance)")
    print("2. codellama-13b-instruct (Best for technical papers)")
    print("3. mistral-7b-instruct (Fastest, good for batch processing)")
    
    while True:
        choice = input("Enter choice (1-3) or press Enter for default [1]: ").strip()
        if not choice:
            choice = "1"
        
        if choice == "1":
            return "llama-3.1-8b-instruct"
        elif choice == "2":
            return "codellama-13b-instruct"
        elif choice == "3":
            return "mistral-7b-instruct"
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def check_docker():
    """Check if Docker is available and working."""
    print("\n🐳 Checking Docker...")
    
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is installed")
            
            # Check if Docker daemon is running
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Docker daemon is running")
                return True
            else:
                print("❌ Docker daemon is not running")
                print("Please start Docker and try again")
                return False
        else:
            print("❌ Docker not found")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        print("Please install Docker from: https://docs.docker.com/get-docker/")
        return False

def deploy_nim_service(model_name):
    """Deploy NIM service."""
    print(f"\n🚀 Deploying NIM service with {model_name}...")
    print("This may take a few minutes to download the model...")
    
    try:
        from deploy_nim_service import NIMDeploymentManager
        
        manager = NIMDeploymentManager()
        success = manager.deploy_model(model_name, detached=True)
        
        if success:
            print("✅ NIM service deployed successfully!")
            print("🌐 Service running at: http://localhost:8000")
            return True
        else:
            print("❌ Failed to deploy NIM service")
            return False
            
    except ImportError:
        print("❌ NIM deployment modules not found")
        return False
    except Exception as e:
        print(f"❌ Error deploying NIM service: {e}")
        return False

def test_nim_connection():
    """Test connection to NIM service."""
    print("\n🧪 Testing NIM connection...")
    
    try:
        from nvidia_nim_analyzer import NVIDIANIMAnalyzer, NIMConfig
        
        config = NIMConfig()
        analyzer = NVIDIANIMAnalyzer(config)
        
        if analyzer.test_connection():
            print("✅ NIM service is ready!")
            return True
        else:
            print("❌ Cannot connect to NIM service")
            return False
            
    except Exception as e:
        print(f"❌ Error testing connection: {e}")
        return False

def run_sample_analysis():
    """Run a sample analysis."""
    print("\n📄 Running sample analysis...")
    
    sample_paper = """
    Title: Attention Is All You Need
    
    Abstract: The dominant sequence transduction models are based on complex recurrent
    or convolutional neural networks that include an encoder and a decoder. The best
    performing models also connect the encoder and decoder through an attention
    mechanism. We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and convolutions
    entirely.
    
    The Transformer allows for significantly more parallelization and can reach a new
    state of the art in translation quality after being trained for as little as twelve hours
    on eight P100 GPUs.
    """
    
    try:
        from nvidia_nim_analyzer import NVIDIANIMAnalyzer, NIMConfig
        
        config = NIMConfig()
        analyzer = NVIDIANIMAnalyzer(config)
        
        result = analyzer.analyze_paper_text(
            sample_paper,
            paper_id="transformer_sample",
            title="Attention Is All You Need"
        )
        
        print("✅ Sample analysis completed!")
        print(f"📊 Summary: {result.summary}")
        print(f"🎯 Confidence: {result.confidence_score:.2f}")
        print(f"⏱️  Processing time: {result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample analysis failed: {e}")
        return False

def analyze_existing_papers():
    """Analyze papers from existing pipeline."""
    print("\n📚 Analyzing papers from your existing pipeline...")
    
    # Check for existing processed data
    data_dirs = [
        'bert_training_output/processed_data',
        'downloaded_papers',
        'processed_data'
    ]
    
    found_data = False
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            print(f"✅ Found data directory: {data_dir}")
            found_data = True
            break
    
    if not found_data:
        print("⚠️  No processed paper data found")
        print("Please run your existing pipeline first to generate paper data")
        return False
    
    try:
        # Run the integrated pipeline with a small sample
        cmd = [
            sys.executable, 'nim_paper_pipeline.py',
            '--data-dir', data_dir,
            '--max-papers', '3',  # Analyze just 3 papers for demo
            '--batch-size', '1'
        ]
        
        print("Running command:", ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Paper analysis completed!")
            print("Check the nim_analysis_output directory for results")
            return True
        else:
            print("❌ Paper analysis failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running paper analysis: {e}")
        return False

def print_next_steps():
    """Print next steps and usage information."""
    print("\n🎉 Setup Complete!")
    print("=" * 40)
    print("Your NVIDIA NIM integration is ready to use!")
    print()
    print("📋 Next Steps:")
    print("1. Analyze all your papers:")
    print("   python nim_paper_pipeline.py --max-papers 50")
    print()
    print("2. Use different models:")
    print("   python nim_paper_pipeline.py --model codellama-13b-instruct")
    print()
    print("3. Deploy different models:")
    print("   python deploy_nim_service.py deploy --model mistral-7b-instruct")
    print()
    print("4. Check service status:")
    print("   python deploy_nim_service.py status")
    print()
    print("📁 Output files are saved in: nim_analysis_output/")
    print("📊 Results include JSON reports and CSV summaries")
    print()
    print("💡 Tips:")
    print("- Use --max-papers to limit analysis for testing")
    print("- Larger models (codellama-13b) are better for technical papers")
    print("- Smaller models (mistral-7b) are faster for batch processing")

def main():
    """Main quick start function."""
    print_header()
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed")
        return
    
    # Step 2: Setup environment
    if not setup_environment():
        print("\n❌ Environment setup failed")
        return
    
    # Step 3: Check Docker
    if not check_docker():
        print("\n❌ Docker check failed")
        return
    
    # Step 4: Choose model
    model_name = choose_model()
    print(f"Selected model: {model_name}")
    
    # Step 5: Deploy NIM service
    deploy_choice = input("\nDeploy NIM service now? (y/n) [y]: ").strip().lower()
    if deploy_choice in ['', 'y', 'yes']:
        if not deploy_nim_service(model_name):
            print("\n❌ NIM deployment failed")
            return
        
        # Step 6: Test connection
        time.sleep(10)  # Give service time to start
        if not test_nim_connection():
            print("\n❌ Connection test failed")
            return
        
        # Step 7: Run sample analysis
        sample_choice = input("\nRun sample analysis? (y/n) [y]: ").strip().lower()
        if sample_choice in ['', 'y', 'yes']:
            run_sample_analysis()
        
        # Step 8: Analyze existing papers
        existing_choice = input("\nAnalyze papers from existing pipeline? (y/n) [y]: ").strip().lower()
        if existing_choice in ['', 'y', 'yes']:
            analyze_existing_papers()
    
    # Step 9: Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
