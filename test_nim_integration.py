#!/usr/bin/env python3
"""
Test script for NVIDIA NIM integration.

This script provides simple tests to verify the NIM integration is working correctly.
Run this after setting up NIM to ensure everything is functioning properly.

Author: Generated for research paper analysis pipeline
"""

import json
import time
from pathlib import Path

def test_nim_analyzer():
    """Test basic NIM analyzer functionality."""
    print("🧪 Testing NIM Analyzer...")
    
    try:
        from nvidia_nim_analyzer import NVIDIANIMAnalyzer, NIMConfig
        
        # Create configuration
        config = NIMConfig(
            model_name="llama-3.1-8b-instruct",
            max_tokens=1024,
            temperature=0.7
        )
        
        # Initialize analyzer
        analyzer = NVIDIANIMAnalyzer(config)
        
        # Test connection
        if not analyzer.test_connection():
            print("❌ NIM service not available. Make sure it's running:")
            print("   python deploy_nim_service.py deploy --model llama-3.1-8b-instruct")
            return False
        
        print("✅ NIM service connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Error testing NIM analyzer: {e}")
        return False

def test_sample_analysis():
    """Test analysis with a sample paper."""
    print("\n📄 Testing sample paper analysis...")
    
    sample_paper = """
    Title: Transformer Networks for Natural Language Processing
    
    Abstract: This paper presents a comprehensive study of transformer networks
    and their applications in natural language processing tasks. We demonstrate
    that attention mechanisms can effectively capture long-range dependencies
    in sequential data without the need for recurrent connections.
    
    Introduction: Natural language processing has been revolutionized by the
    introduction of transformer architectures. These models rely entirely on
    attention mechanisms to draw global dependencies between input and output.
    
    Methodology: We implemented a standard transformer architecture with
    multi-head self-attention layers and position-wise feed-forward networks.
    The model was trained on a large corpus of text data using standard
    cross-entropy loss.
    
    Results: Our experiments show that the transformer model achieves
    state-of-the-art performance on several NLP benchmarks, including
    machine translation and text classification tasks.
    
    Conclusion: Transformer networks represent a significant advancement
    in sequence modeling and offer promising directions for future research
    in natural language understanding.
    """
    
    try:
        from nvidia_nim_analyzer import NVIDIANIMAnalyzer, NIMConfig
        
        config = NIMConfig()
        analyzer = NVIDIANIMAnalyzer(config)
        
        # Analyze the sample paper
        result = analyzer.analyze_paper_text(
            sample_paper,
            paper_id="test_transformer_paper",
            title="Transformer Networks for Natural Language Processing"
        )
        
        # Print results
        print("✅ Analysis completed successfully!")
        print(f"📊 Summary: {result.summary}")
        print(f"🔍 Key Findings ({len(result.key_findings)}): {result.key_findings[:3]}")
        print(f"⚠️  Limitations ({len(result.limitations)}): {result.limitations[:2]}")
        print(f"🚀 Future Work ({len(result.future_work)}): {result.future_work[:2]}")
        print(f"📈 Sentiment Score: {result.sentiment_score:.2f}")
        print(f"🎯 Confidence Score: {result.confidence_score:.2f}")
        print(f"⏱️  Processing Time: {result.processing_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in sample analysis: {e}")
        return False

def test_batch_analysis():
    """Test batch analysis with multiple papers."""
    print("\n📚 Testing batch analysis...")
    
    papers = [
        {
            "text": "This paper explores deep learning applications in computer vision...",
            "id": "cv_paper_1",
            "title": "Deep Learning for Computer Vision"
        },
        {
            "text": "We present a novel approach to reinforcement learning using neural networks...",
            "id": "rl_paper_1", 
            "title": "Neural Reinforcement Learning"
        },
        {
            "text": "This study investigates the use of attention mechanisms in sequence modeling...",
            "id": "attention_paper_1",
            "title": "Attention Mechanisms in Sequence Models"
        }
    ]
    
    try:
        from nvidia_nim_analyzer import NVIDIANIMAnalyzer, NIMConfig
        
        config = NIMConfig()
        analyzer = NVIDIANIMAnalyzer(config)
        
        # Analyze batch
        results = analyzer.analyze_paper_batch(papers, max_concurrent=2)
        
        print(f"✅ Batch analysis completed!")
        print(f"📊 Papers processed: {len(results)}/{len(papers)}")
        
        if results:
            avg_confidence = sum(r.confidence_score for r in results) / len(results)
            avg_time = sum(r.processing_time for r in results) / len(results)
            print(f"🎯 Average confidence: {avg_confidence:.2f}")
            print(f"⏱️  Average processing time: {avg_time:.2f}s")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error in batch analysis: {e}")
        return False

def test_deployment_manager():
    """Test deployment manager functionality."""
    print("\n🚀 Testing deployment manager...")
    
    try:
        from deploy_nim_service import NIMDeploymentManager
        
        manager = NIMDeploymentManager()
        
        # Test prerequisites check
        print("Checking prerequisites...")
        prereqs_ok = manager.check_prerequisites()
        
        # List available models
        print("Available models:")
        models = manager.list_available_models()
        for name, info in models.items():
            print(f"  - {name}: {info['description']}")
        
        # Check running containers
        containers = manager.list_running_containers()
        print(f"Running NIM containers: {len(containers)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing deployment manager: {e}")
        return False

def test_pipeline_integration():
    """Test the full pipeline integration."""
    print("\n🔗 Testing pipeline integration...")
    
    # Check if processed data exists
    data_paths = [
        "bert_training_output/processed_data",
        "downloaded_papers",
        "processed_data"
    ]
    
    found_data = None
    for path in data_paths:
        if Path(path).exists():
            found_data = path
            break
    
    if not found_data:
        print("⚠️  No processed data found. Creating sample data for testing...")
        # Create sample data
        sample_data = [
            {
                "text": "Sample research paper about machine learning and its applications in data science...",
                "id": "sample_1",
                "title": "Machine Learning in Data Science"
            }
        ]
        
        test_data_path = Path("test_data.json")
        with open(test_data_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        found_data = str(test_data_path)
        print(f"✅ Created sample data: {found_data}")
    
    try:
        from nim_paper_pipeline import PaperPipelineIntegration
        from nvidia_nim_analyzer import NIMConfig
        
        # Initialize pipeline
        config = NIMConfig()
        pipeline = PaperPipelineIntegration(config, "test_output")
        
        # Load papers
        if found_data.endswith('.json'):
            papers = pipeline.load_processed_papers(found_data)
        else:
            papers = pipeline.extract_paper_text_from_processed_data(found_data)
        
        if papers:
            print(f"✅ Loaded {len(papers)} papers for testing")
            
            # Analyze a small sample
            sample_papers = papers[:2]  # Just 2 papers for testing
            results = pipeline.analyze_papers_with_nim(sample_papers, max_papers=2, batch_size=1)
            
            if results:
                print(f"✅ Pipeline analysis successful: {len(results)} papers analyzed")
                
                # Generate report
                report = pipeline.generate_comprehensive_report(results)
                print(f"📊 Report generated with {len(report)} sections")
                
                return True
            else:
                print("❌ No successful analyses in pipeline test")
                return False
        else:
            print("❌ No papers loaded for pipeline test")
            return False
            
    except Exception as e:
        print(f"❌ Error testing pipeline integration: {e}")
        return False
    finally:
        # Clean up test data
        test_file = Path("test_data.json")
        if test_file.exists():
            test_file.unlink()

def main():
    """Run all tests."""
    print("🧪 NVIDIA NIM Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("NIM Analyzer Basic Test", test_nim_analyzer),
        ("Sample Analysis Test", test_sample_analysis),
        ("Batch Analysis Test", test_batch_analysis),
        ("Deployment Manager Test", test_deployment_manager),
        ("Pipeline Integration Test", test_pipeline_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        
        start_time = time.time()
        success = test_func()
        duration = time.time() - start_time
        
        results.append((test_name, success, duration))
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} ({duration:.2f}s)")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration:.2f}s)")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your NIM integration is working correctly.")
        print("\n💡 Next steps:")
        print("1. Run full analysis: python nim_paper_pipeline.py")
        print("2. Try different models: python deploy_nim_service.py list")
        print("3. Check the README_NIM_INTEGRATION.md for advanced usage")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("💡 Common solutions:")
        print("1. Ensure NIM service is running: python deploy_nim_service.py deploy --model llama-3.1-8b-instruct")
        print("2. Check Docker and GPU setup: python deploy_nim_service.py check")
        print("3. Verify NGC API key is set: echo $NGC_API_KEY")

if __name__ == "__main__":
    main()
