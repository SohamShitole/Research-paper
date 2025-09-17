#!/usr/bin/env python3
"""
NVIDIA NIM Service Deployment Script

This script helps deploy and manage NVIDIA NIM services for research paper analysis.
It provides utilities to start, stop, and manage NIM containers with different models.

Author: Generated for research paper analysis pipeline
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NIMDeploymentManager:
    """Manager for NVIDIA NIM service deployment."""
    
    def __init__(self, cache_path: str = None):
        """Initialize the deployment manager."""
        self.cache_path = cache_path or os.path.expanduser("~/nim_cache")
        self.containers = {}
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure NIM cache directory exists."""
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        os.chmod(self.cache_path, 0o777)
        logger.info(f"NIM cache directory: {self.cache_path}")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        checks = []
        
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                checks.append(("Docker", True, result.stdout.strip()))
            else:
                checks.append(("Docker", False, "Not installed"))
        except FileNotFoundError:
            checks.append(("Docker", False, "Not found"))
        
        # Check NVIDIA Docker runtime
        try:
            result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 
                                   'nvidia/cuda:11.8-base-ubuntu20.04', 
                                   'nvidia-smi'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                checks.append(("NVIDIA Docker", True, "GPU access available"))
            else:
                checks.append(("NVIDIA Docker", False, "GPU access failed"))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            checks.append(("NVIDIA Docker", False, "Cannot test GPU access"))
        
        # Check NGC API Key
        ngc_key = os.getenv('NGC_API_KEY')
        if ngc_key:
            checks.append(("NGC API Key", True, f"Set (length: {len(ngc_key)})"))
        else:
            checks.append(("NGC API Key", False, "Not set in environment"))
        
        # Print results
        print("\n🔍 Prerequisites Check:")
        print("-" * 50)
        all_good = True
        for name, status, details in checks:
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {name}: {details}")
            if not status:
                all_good = False
        
        if not all_good:
            print("\n⚠️  Some prerequisites are missing. Please fix them before proceeding.")
            self._print_setup_instructions()
        
        return all_good
    
    def _print_setup_instructions(self):
        """Print setup instructions for missing prerequisites."""
        print("\n📋 Setup Instructions:")
        print("-" * 30)
        print("1. Install Docker: https://docs.docker.com/get-docker/")
        print("2. Install NVIDIA Container Toolkit:")
        print("   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html")
        print("3. Get NGC API Key from: https://ngc.nvidia.com/setup/api-key")
        print("4. Set environment variable: export NGC_API_KEY=your_key_here")
    
    def list_available_models(self) -> Dict[str, Dict]:
        """List available NIM models."""
        models = {
            'llama-3.1-8b-instruct': {
                'image': 'nvcr.io/nim/meta/llama-3.1-8b-instruct:1.0.0',
                'description': 'Meta Llama 3.1 8B Instruct - Balanced performance',
                'memory_req': '16GB',
                'recommended_for': 'General research paper analysis'
            },
            'codellama-13b-instruct': {
                'image': 'nvcr.io/nim/meta/codellama-13b-instruct:1.2.2',
                'description': 'CodeLlama 13B Instruct - Technical content specialist',
                'memory_req': '24GB',
                'recommended_for': 'Technical papers, code analysis'
            },
            'mistral-7b-instruct': {
                'image': 'nvcr.io/nim/mistralai/mistral-7b-instruct:1.0.0',
                'description': 'Mistral 7B Instruct - Fast and efficient',
                'memory_req': '14GB',
                'recommended_for': 'Quick analysis, batch processing'
            }
        }
        return models
    
    def deploy_model(self, model_name: str, port: int = 8000, 
                    detached: bool = True, gpu_memory: str = "16GB") -> bool:
        """
        Deploy a NIM model.
        
        Args:
            model_name: Name of the model to deploy
            port: Port to run the service on
            detached: Run in detached mode
            gpu_memory: Shared memory size
            
        Returns:
            True if deployment successful, False otherwise
        """
        models = self.list_available_models()
        if model_name not in models:
            logger.error(f"Model {model_name} not available. Choose from: {list(models.keys())}")
            return False
        
        model_info = models[model_name]
        container_name = f"nim-{model_name.replace('/', '-').replace(':', '-')}"
        
        # Check if container already running
        if self._is_container_running(container_name):
            logger.info(f"Container {container_name} already running")
            return True
        
        # Build Docker command
        docker_cmd = [
            'docker', 'run',
            '--name', container_name,
            '--gpus', 'all',
            '--shm-size', gpu_memory,
            '-p', f'{port}:8000',
            '-e', f'NGC_API_KEY={os.getenv("NGC_API_KEY")}',
            '-v', f'{self.cache_path}:/opt/nim/.cache'
        ]
        
        if detached:
            docker_cmd.append('-d')
        else:
            docker_cmd.extend(['--rm', '-it'])
        
        docker_cmd.append(model_info['image'])
        
        logger.info(f"Deploying {model_name} on port {port}...")
        logger.info(f"Command: {' '.join(docker_cmd)}")
        
        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Successfully deployed {model_name}")
                if detached:
                    self.containers[model_name] = {
                        'container_name': container_name,
                        'port': port,
                        'status': 'running'
                    }
                    # Wait for service to be ready
                    self._wait_for_service(port)
                return True
            else:
                logger.error(f"❌ Failed to deploy {model_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Deployment error: {e}")
            return False
    
    def _is_container_running(self, container_name: str) -> bool:
        """Check if a container is running."""
        try:
            result = subprocess.run(['docker', 'ps', '--filter', f'name={container_name}', 
                                   '--format', '{{.Names}}'], 
                                  capture_output=True, text=True)
            return container_name in result.stdout
        except Exception:
            return False
    
    def _wait_for_service(self, port: int, timeout: int = 300):
        """Wait for the NIM service to be ready."""
        import time
        import requests
        
        url = f"http://localhost:{port}/v1/models"
        start_time = time.time()
        
        print(f"⏳ Waiting for service on port {port} to be ready...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Service ready on port {port}")
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(10)
            print(".", end="", flush=True)
        
        logger.warning(f"⚠️  Service on port {port} not ready after {timeout}s")
        return False
    
    def stop_model(self, model_name: str) -> bool:
        """Stop a deployed model."""
        container_name = f"nim-{model_name.replace('/', '-').replace(':', '-')}"
        
        try:
            result = subprocess.run(['docker', 'stop', container_name], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Stopped {model_name}")
                if model_name in self.containers:
                    del self.containers[model_name]
                return True
            else:
                logger.error(f"❌ Failed to stop {model_name}: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Stop error: {e}")
            return False
    
    def list_running_containers(self) -> List[Dict]:
        """List running NIM containers."""
        try:
            result = subprocess.run(['docker', 'ps', '--filter', 'name=nim-', 
                                   '--format', '{{.Names}}\t{{.Ports}}\t{{.Status}}'], 
                                  capture_output=True, text=True)
            
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        containers.append({
                            'name': parts[0],
                            'ports': parts[1],
                            'status': parts[2]
                        })
            
            return containers
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []
    
    def cleanup_all(self) -> bool:
        """Stop and remove all NIM containers."""
        containers = self.list_running_containers()
        success = True
        
        for container in containers:
            try:
                # Stop container
                subprocess.run(['docker', 'stop', container['name']], 
                             capture_output=True, text=True)
                # Remove container
                subprocess.run(['docker', 'rm', container['name']], 
                             capture_output=True, text=True)
                logger.info(f"Cleaned up {container['name']}")
            except Exception as e:
                logger.error(f"Failed to cleanup {container['name']}: {e}")
                success = False
        
        return success


def main():
    """Main function for NIM deployment management."""
    parser = argparse.ArgumentParser(description='NVIDIA NIM Deployment Manager')
    parser.add_argument('action', choices=['check', 'list', 'deploy', 'stop', 'status', 'cleanup'],
                       help='Action to perform')
    parser.add_argument('--model', type=str, help='Model name for deploy/stop actions')
    parser.add_argument('--port', type=int, default=8000, help='Port for deployment')
    parser.add_argument('--cache-path', type=str, help='Custom cache path')
    parser.add_argument('--gpu-memory', type=str, default='16GB', help='GPU shared memory size')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = NIMDeploymentManager(args.cache_path)
    
    print("🚀 NVIDIA NIM Deployment Manager")
    print("=" * 40)
    
    if args.action == 'check':
        manager.check_prerequisites()
    
    elif args.action == 'list':
        models = manager.list_available_models()
        print("\n📋 Available Models:")
        print("-" * 40)
        for name, info in models.items():
            print(f"🤖 {name}")
            print(f"   Description: {info['description']}")
            print(f"   Memory Req: {info['memory_req']}")
            print(f"   Best for: {info['recommended_for']}")
            print()
    
    elif args.action == 'deploy':
        if not args.model:
            print("❌ Please specify --model for deployment")
            sys.exit(1)
        
        if not manager.check_prerequisites():
            sys.exit(1)
        
        success = manager.deploy_model(args.model, args.port, gpu_memory=args.gpu_memory)
        if success:
            print(f"\n✅ {args.model} deployed successfully on port {args.port}")
            print(f"🌐 API endpoint: http://localhost:{args.port}/v1")
            print("\n🧪 Test with:")
            print(f"python nvidia_nim_analyzer.py")
        else:
            print(f"❌ Failed to deploy {args.model}")
            sys.exit(1)
    
    elif args.action == 'stop':
        if not args.model:
            print("❌ Please specify --model to stop")
            sys.exit(1)
        
        success = manager.stop_model(args.model)
        if not success:
            sys.exit(1)
    
    elif args.action == 'status':
        containers = manager.list_running_containers()
        if containers:
            print("\n🏃 Running NIM Services:")
            print("-" * 40)
            for container in containers:
                print(f"📦 {container['name']}")
                print(f"   Ports: {container['ports']}")
                print(f"   Status: {container['status']}")
                print()
        else:
            print("\n😴 No NIM services currently running")
    
    elif args.action == 'cleanup':
        print("\n🧹 Cleaning up all NIM containers...")
        success = manager.cleanup_all()
        if success:
            print("✅ Cleanup completed")
        else:
            print("❌ Some cleanup operations failed")


if __name__ == "__main__":
    main()
