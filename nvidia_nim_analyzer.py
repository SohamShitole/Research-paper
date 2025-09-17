#!/usr/bin/env python3
"""
NVIDIA NIM Integration for Research Paper Analysis

This module provides an interface to use NVIDIA NIM's open source models
for analyzing research papers as an alternative or complement to the BERT pipeline.

Supported models:
- Llama-3.1-8B-Instruct
- CodeLlama-13B-Instruct
- Mistral-7B-Instruct
- Yi-34B-Chat

Author: Generated for research paper analysis pipeline
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import time

try:
    import openai
    from openai import OpenAI
except ImportError:
    print("OpenAI library not found. Install with: pip install openai")
    raise

try:
    import pandas as pd
except ImportError:
    print("Pandas not found. Install with: pip install pandas")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NIMConfig:
    """Configuration for NVIDIA NIM API."""
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "none"  # NIM uses 'none' as default
    model_name: str = "llama-3.1-8b-instruct"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 60


@dataclass
class PaperAnalysisResult:
    """Result structure for paper analysis."""
    paper_id: str
    title: str
    summary: str
    key_findings: List[str]
    methodology: str
    limitations: List[str]
    future_work: List[str]
    sentiment_score: float
    confidence_score: float
    processing_time: float
    model_used: str


class NVIDIANIMAnalyzer:
    """
    NVIDIA NIM-based analyzer for research papers.
    
    This class provides methods to analyze research papers using 
    NVIDIA NIM's open source language models.
    """
    
    def __init__(self, config: NIMConfig = None):
        """Initialize the NIM analyzer."""
        self.config = config or NIMConfig()
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        """Initialize the OpenAI client for NIM."""
        try:
            self.client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            logger.info(f"Initialized NIM client with base URL: {self.config.base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize NIM client: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to NIM service."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                max_tokens=50,
                temperature=0.1
            )
            logger.info("✅ NIM connection test successful")
            logger.info(f"Model response: {response.choices[0].message.content}")
            return True
        except Exception as e:
            logger.error(f"❌ NIM connection test failed: {e}")
            return False
    
    def analyze_paper_text(self, paper_text: str, paper_id: str = None, title: str = None) -> PaperAnalysisResult:
        """
        Analyze a research paper using NIM.
        
        Args:
            paper_text: Full text of the research paper
            paper_id: Unique identifier for the paper
            title: Title of the paper
            
        Returns:
            PaperAnalysisResult object with analysis results
        """
        start_time = time.time()
        
        # Create analysis prompt
        analysis_prompt = self._create_analysis_prompt(paper_text, title)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            analysis_text = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            # Parse the structured response
            result = self._parse_analysis_response(
                analysis_text, 
                paper_id or "unknown", 
                title or "Unknown Title",
                processing_time
            )
            
            logger.info(f"Successfully analyzed paper: {paper_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze paper {paper_id}: {e}")
            # Return empty result on failure
            return PaperAnalysisResult(
                paper_id=paper_id or "unknown",
                title=title or "Unknown Title",
                summary="Analysis failed",
                key_findings=[],
                methodology="Unknown",
                limitations=["Analysis failed"],
                future_work=[],
                sentiment_score=0.0,
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                model_used=self.config.model_name
            )
    
    def analyze_paper_batch(self, papers: List[Dict[str, str]], max_concurrent: int = 3) -> List[PaperAnalysisResult]:
        """
        Analyze multiple papers concurrently.
        
        Args:
            papers: List of dictionaries with 'text', 'id', and 'title' keys
            max_concurrent: Maximum number of concurrent analyses
            
        Returns:
            List of PaperAnalysisResult objects
        """
        async def analyze_single(paper_data):
            return self.analyze_paper_text(
                paper_data['text'],
                paper_data.get('id'),
                paper_data.get('title')
            )
        
        async def analyze_batch():
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def bounded_analyze(paper_data):
                async with semaphore:
                    return analyze_single(paper_data)
            
            tasks = [bounded_analyze(paper) for paper in papers]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Run the async batch processing
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(analyze_batch())
        
        # Filter out exceptions and return successful results
        successful_results = [r for r in results if isinstance(r, PaperAnalysisResult)]
        logger.info(f"Successfully analyzed {len(successful_results)}/{len(papers)} papers")
        
        return successful_results
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for paper analysis."""
        return """You are an expert research paper analyst. Your task is to analyze academic papers and provide structured insights. 

Please analyze the given research paper and provide your response in the following JSON format:

{
    "summary": "A concise 2-3 sentence summary of the paper",
    "key_findings": ["finding 1", "finding 2", "finding 3"],
    "methodology": "Brief description of the methodology used",
    "limitations": ["limitation 1", "limitation 2"],
    "future_work": ["future direction 1", "future direction 2"],
    "sentiment_score": 0.8,
    "confidence_score": 0.9
}

Where:
- sentiment_score: 0-1 scale indicating how positive/promising the research is
- confidence_score: 0-1 scale indicating your confidence in the analysis

Be thorough but concise. Focus on the most important aspects of the research."""
    
    def _create_analysis_prompt(self, paper_text: str, title: str = None) -> str:
        """Create the analysis prompt for a paper."""
        prompt = f"""Please analyze this research paper:

Title: {title or 'Not provided'}

Paper Content:
{paper_text[:8000]}  # Limit text to avoid token limits

Please provide a structured analysis following the JSON format specified in the system prompt."""
        
        return prompt
    
    def _parse_analysis_response(self, response_text: str, paper_id: str, title: str, processing_time: float) -> PaperAnalysisResult:
        """Parse the NIM response into a structured result."""
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = response_text[json_start:json_end]
                parsed_data = json.loads(json_text)
                
                return PaperAnalysisResult(
                    paper_id=paper_id,
                    title=title,
                    summary=parsed_data.get('summary', 'No summary provided'),
                    key_findings=parsed_data.get('key_findings', []),
                    methodology=parsed_data.get('methodology', 'No methodology described'),
                    limitations=parsed_data.get('limitations', []),
                    future_work=parsed_data.get('future_work', []),
                    sentiment_score=float(parsed_data.get('sentiment_score', 0.5)),
                    confidence_score=float(parsed_data.get('confidence_score', 0.5)),
                    processing_time=processing_time,
                    model_used=self.config.model_name
                )
            else:
                # Fallback: create result from unstructured text
                return self._create_fallback_result(response_text, paper_id, title, processing_time)
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse structured response for {paper_id}: {e}")
            return self._create_fallback_result(response_text, paper_id, title, processing_time)
    
    def _create_fallback_result(self, response_text: str, paper_id: str, title: str, processing_time: float) -> PaperAnalysisResult:
        """Create a fallback result when structured parsing fails."""
        return PaperAnalysisResult(
            paper_id=paper_id,
            title=title,
            summary=response_text[:500] + "..." if len(response_text) > 500 else response_text,
            key_findings=["Analysis available in summary"],
            methodology="See summary for details",
            limitations=["Structured parsing failed"],
            future_work=["See summary for details"],
            sentiment_score=0.5,
            confidence_score=0.3,  # Lower confidence for unstructured results
            processing_time=processing_time,
            model_used=self.config.model_name
        )
    
    def compare_with_bert_results(self, nim_results: List[PaperAnalysisResult], bert_results_path: str) -> Dict[str, Any]:
        """
        Compare NIM results with existing BERT results.
        
        Args:
            nim_results: List of NIM analysis results
            bert_results_path: Path to BERT results file
            
        Returns:
            Dictionary with comparison metrics
        """
        try:
            # Load BERT results (assuming they're in JSON format)
            with open(bert_results_path, 'r') as f:
                bert_data = json.load(f)
            
            comparison = {
                'nim_papers_analyzed': len(nim_results),
                'bert_papers_analyzed': len(bert_data),
                'average_nim_processing_time': sum(r.processing_time for r in nim_results) / len(nim_results),
                'average_nim_confidence': sum(r.confidence_score for r in nim_results) / len(nim_results),
                'model_used': self.config.model_name,
                'comparison_timestamp': time.time()
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare with BERT results: {e}")
            return {'error': str(e)}
    
    def save_results(self, results: List[PaperAnalysisResult], output_path: str):
        """Save analysis results to JSON file."""
        results_data = []
        for result in results:
            results_data.append({
                'paper_id': result.paper_id,
                'title': result.title,
                'summary': result.summary,
                'key_findings': result.key_findings,
                'methodology': result.methodology,
                'limitations': result.limitations,
                'future_work': result.future_work,
                'sentiment_score': result.sentiment_score,
                'confidence_score': result.confidence_score,
                'processing_time': result.processing_time,
                'model_used': result.model_used
            })
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Saved {len(results)} analysis results to {output_path}")
    
    def create_summary_report(self, results: List[PaperAnalysisResult]) -> Dict[str, Any]:
        """Create a summary report of the analysis results."""
        if not results:
            return {'error': 'No results to summarize'}
        
        total_papers = len(results)
        avg_confidence = sum(r.confidence_score for r in results) / total_papers
        avg_sentiment = sum(r.sentiment_score for r in results) / total_papers
        avg_processing_time = sum(r.processing_time for r in results) / total_papers
        
        # Count findings and limitations
        total_findings = sum(len(r.key_findings) for r in results)
        total_limitations = sum(len(r.limitations) for r in results)
        
        return {
            'total_papers_analyzed': total_papers,
            'average_confidence_score': round(avg_confidence, 3),
            'average_sentiment_score': round(avg_sentiment, 3),
            'average_processing_time_seconds': round(avg_processing_time, 2),
            'total_key_findings': total_findings,
            'total_limitations_identified': total_limitations,
            'model_used': self.config.model_name,
            'analysis_timestamp': time.time(),
            'papers_with_high_confidence': len([r for r in results if r.confidence_score > 0.8]),
            'papers_with_positive_sentiment': len([r for r in results if r.sentiment_score > 0.6])
        }


# Available NIM model configurations
NIM_MODELS = {
    'llama-3.1-8b-instruct': {
        'name': 'llama-3.1-8b-instruct',
        'description': 'Meta Llama 3.1 8B Instruct - Good balance of speed and capability',
        'image': 'nvcr.io/nim/meta/llama-3.1-8b-instruct:1.0.0'
    },
    'codellama-13b-instruct': {
        'name': 'codellama-13b-instruct', 
        'description': 'CodeLlama 13B Instruct - Excellent for technical content',
        'image': 'nvcr.io/nim/meta/codellama-13b-instruct:1.2.2'
    },
    'mistral-7b-instruct': {
        'name': 'mistral-7b-instruct',
        'description': 'Mistral 7B Instruct - Fast and efficient',
        'image': 'nvcr.io/nim/mistralai/mistral-7b-instruct:1.0.0'
    }
}


def create_nim_config(model_name: str = 'llama-3.1-8b-instruct', 
                      base_url: str = 'http://localhost:8000/v1',
                      max_tokens: int = 2048) -> NIMConfig:
    """Create a NIM configuration with the specified model."""
    if model_name not in NIM_MODELS:
        logger.warning(f"Model {model_name} not in known models. Using default.")
        model_name = 'llama-3.1-8b-instruct'
    
    return NIMConfig(
        base_url=base_url,
        model_name=model_name,
        max_tokens=max_tokens
    )


def main():
    """Example usage of the NVIDIA NIM analyzer."""
    print("🚀 NVIDIA NIM Research Paper Analyzer")
    print("=" * 50)
    
    # Create configuration
    config = create_nim_config('llama-3.1-8b-instruct')
    
    # Initialize analyzer
    analyzer = NVIDIANIMAnalyzer(config)
    
    # Test connection
    if not analyzer.test_connection():
        print("❌ Cannot connect to NIM service. Make sure it's running.")
        print("\nTo start NIM service, run:")
        print("docker run --rm --gpus all --shm-size=16GB --network=host \\")
        print("  -e NGC_API_KEY=$NGC_API_KEY \\")
        print("  -v $NIM_CACHE_PATH:/opt/nim/.cache \\")
        print("  nvcr.io/nim/meta/llama-3.1-8b-instruct:1.0.0")
        return
    
    # Example analysis
    sample_paper = """
    Title: Attention Is All You Need
    
    Abstract: The dominant sequence transduction models are based on complex recurrent
    or convolutional neural networks that include an encoder and a decoder. The best
    performing models also connect the encoder and decoder through an attention
    mechanism. We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and convolutions
    entirely.
    
    Introduction: Recurrent neural networks, long short-term memory and gated
    recurrent neural networks in particular, have been firmly established as state
    of the art approaches in sequence modeling and transduction problems...
    """
    
    print("\n📄 Analyzing sample paper...")
    result = analyzer.analyze_paper_text(
        sample_paper, 
        paper_id="transformer_paper",
        title="Attention Is All You Need"
    )
    
    print(f"\n✅ Analysis completed in {result.processing_time:.2f} seconds")
    print(f"📊 Summary: {result.summary}")
    print(f"🔍 Key Findings: {', '.join(result.key_findings[:3])}")
    print(f"📈 Confidence Score: {result.confidence_score:.2f}")
    
    print("\n" + "=" * 50)
    print("🎯 Ready to analyze your research papers with NVIDIA NIM!")


if __name__ == "__main__":
    main()
