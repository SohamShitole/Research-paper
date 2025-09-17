#!/usr/bin/env python3
"""
Setup script for Hugging Face authentication and model caching.

This script helps with:
1. Hugging Face authentication
2. Pre-downloading models for offline use
3. Setting up local model cache
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_huggingface_hub():
    """Install or upgrade huggingface_hub."""
    try:
        subprocess.run(['pip3', 'install', '--upgrade', 'huggingface_hub'], check=True)
        logger.info("✅ huggingface_hub installed/upgraded successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install huggingface_hub: {e}")
        return False


def login_huggingface():
    """Login to Hugging Face."""
    logger.info("🔑 Hugging Face Authentication")
    logger.info("You have two options:")
    logger.info("1. Login with your Hugging Face token")
    logger.info("2. Use anonymous access (may have limitations)")
    
    choice = input("\nChoose (1 for login, 2 for anonymous): ").strip()
    
    if choice == "1":
        logger.info("\n📝 To get a token:")
        logger.info("1. Go to https://huggingface.co/settings/tokens")
        logger.info("2. Create a new token (read permission is sufficient)")
        logger.info("3. Copy the token")
        
        token = input("\nEnter your Hugging Face token (or press Enter to skip): ").strip()
        
        if token:
            try:
                # Set environment variable
                os.environ['HUGGINGFACE_HUB_TOKEN'] = token
                
                # Try to login using CLI
                result = subprocess.run(['huggingface-cli', 'login', '--token', token], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Successfully logged into Hugging Face!")
                    return True
                else:
                    logger.warning(f"⚠️ CLI login failed: {result.stderr}")
                    logger.info("Token set as environment variable instead")
                    return True
                    
            except FileNotFoundError:
                logger.info("huggingface-cli not found, setting token as environment variable")
                return True
            except Exception as e:
                logger.error(f"❌ Login failed: {e}")
                return False
        else:
            logger.info("Skipping token login")
    
    logger.info("Using anonymous access")
    return True


def download_models_offline():
    """Pre-download models for offline use."""
    logger.info("📥 Pre-downloading models for offline use...")
    
    models_to_download = [
        'bert-base-uncased',
        'distilbert-base-uncased'  # Alternative smaller model
    ]
    
    for model_name in models_to_download:
        logger.info(f"Downloading {model_name}...")
        try:
            # Download tokenizer and model
            from transformers import AutoTokenizer, AutoModel
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            logger.info(f"✅ {model_name} downloaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to download {model_name}: {e}")
            continue
    
    logger.info("✅ Model download completed!")


def setup_offline_mode():
    """Setup for offline mode."""
    logger.info("🔧 Setting up offline mode...")
    
    # Set environment variables for offline mode
    env_vars = {
        'TRANSFORMERS_OFFLINE': '1',
        'HF_DATASETS_OFFLINE': '1'
    }
    
    # Create a script to set these variables
    env_script = """#!/bin/bash
# Hugging Face Offline Mode Environment Variables
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface

echo "🔧 Hugging Face offline mode activated"
echo "Models will be loaded from local cache only"
"""
    
    with open('setup_offline_env.sh', 'w') as f:
        f.write(env_script)
    
    os.chmod('setup_offline_env.sh', 0o755)
    
    logger.info("✅ Offline environment script created: setup_offline_env.sh")
    logger.info("Run: source setup_offline_env.sh to activate offline mode")


def test_model_access():
    """Test if we can access BERT models."""
    logger.info("🧪 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        logger.info("✅ BERT model access successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Model access failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("🤗 Hugging Face Setup for BERT Training")
    print("=" * 60)
    
    # Install/upgrade huggingface_hub
    if not install_huggingface_hub():
        return 1
    
    # Login to Hugging Face
    if not login_huggingface():
        logger.error("Authentication failed")
        return 1
    
    # Test model access
    if test_model_access():
        logger.info("🎉 Setup completed successfully!")
        logger.info("You can now run: python3 quick_start.py --mode quick")
    else:
        logger.info("⚠️ Model access failed. Trying alternative approaches...")
        
        # Offer to download models for offline use
        download_choice = input("\nWould you like to try downloading models for offline use? (y/n): ").strip().lower()
        
        if download_choice == 'y':
            download_models_offline()
            setup_offline_mode()
            
            logger.info("\n📋 Next steps:")
            logger.info("1. Run: source setup_offline_env.sh")
            logger.info("2. Then: python3 quick_start.py --mode quick")
        else:
            logger.info("\n🔧 Alternative solutions:")
            logger.info("1. Check your internet connection")
            logger.info("2. Try using a different model (e.g., distilbert-base-uncased)")
            logger.info("3. Set up Hugging Face authentication properly")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
