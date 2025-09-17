#!/usr/bin/env python3
"""
NVIDIA NIM Integration with Research Paper Pipeline

This script integrates NVIDIA NIM analysis with the existing BERT-based
research paper analysis pipeline, allowing for comparison and enhanced insights.

Author: Generated for research paper analysis pipeline
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import time

# Import existing modules
try:
    from nvidia_nim_analyzer import NVIDIANIMAnalyzer, NIMConfig, PaperAnalysisResult
    from deploy_nim_service import NIMDeploymentManager
except ImportError as e:
    print(f"Error importing NIM modules: {e}")
    print("Make sure nvidia_nim_analyzer.py and deploy_nim_service.py are in the same directory")
    exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PaperPipelineIntegration:
    """Integration class for NVIDIA NIM with existing paper analysis pipeline."""
    
    def __init__(self, nim_config: NIMConfig = None, output_dir: str = "nim_analysis_output"):
        """Initialize the pipeline integration."""
        self.nim_config = nim_config or NIMConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = NVIDIANIMAnalyzer(self.nim_config)
        self.deployment_manager = NIMDeploymentManager()
        
        logger.info(f"Initialized NIM pipeline integration with output dir: {self.output_dir}")
    
    def load_processed_papers(self, data_path: str) -> List[Dict[str, Any]]:
        """
        Load processed papers from the existing pipeline.
        
        Args:
            data_path: Path to processed paper data (JSON format)
            
        Returns:
            List of paper dictionaries
        """
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            papers = []
            if isinstance(data, list):
                papers = data
            elif isinstance(data, dict) and 'papers' in data:
                papers = data['papers']
            else:
                logger.error(f"Unexpected data format in {data_path}")
                return []
            
            logger.info(f"Loaded {len(papers)} papers from {data_path}")
            return papers
            
        except Exception as e:
            logger.error(f"Failed to load papers from {data_path}: {e}")
            return []
    
    def extract_paper_text_from_processed_data(self, processed_data_dir: str) -> List[Dict[str, str]]:
        """
        Extract paper text from the existing processed data directory.
        
        Args:
            processed_data_dir: Path to processed data directory
            
        Returns:
            List of papers with text, id, and title
        """
        papers = []
        processed_path = Path(processed_data_dir)
        
        # Look for different data formats in the processed directory
        data_files = [
            'train_papers.json',
            'val_papers.json', 
            'test_papers.json',
            'bert_train_dataset.json',
            'bert_val_dataset.json',
            'bert_test_dataset.json'
        ]
        
        for data_file in data_files:
            file_path = processed_path / data_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            paper = self._extract_paper_info(item, data_file)
                            if paper:
                                papers.append(paper)
                    elif isinstance(data, dict):
                        paper = self._extract_paper_info(data, data_file)
                        if paper:
                            papers.append(paper)
                    
                    logger.info(f"Extracted papers from {data_file}")
                    
                except Exception as e:
                    logger.warning(f"Could not process {data_file}: {e}")
        
        logger.info(f"Total papers extracted: {len(papers)}")
        return papers
    
    def _extract_paper_info(self, item: Dict, source_file: str) -> Optional[Dict[str, str]]:
        """Extract paper information from a data item."""
        try:
            # Different formats might have different field names
            text_fields = ['text', 'content', 'full_text', 'paper_text', 'abstract', 'body']
            title_fields = ['title', 'paper_title', 'name']
            id_fields = ['id', 'paper_id', 'openreview_id', 'arxiv_id']
            
            # Extract text
            text = None
            for field in text_fields:
                if field in item and item[field]:
                    text = str(item[field])
                    break
            
            # Extract title
            title = None
            for field in title_fields:
                if field in item and item[field]:
                    title = str(item[field])
                    break
            
            # Extract ID
            paper_id = None
            for field in id_fields:
                if field in item and item[field]:
                    paper_id = str(item[field])
                    break
            
            # If we have both text and some identifier, create paper entry
            if text and (paper_id or title):
                return {
                    'text': text,
                    'id': paper_id or f"paper_from_{source_file}_{hash(text) % 10000}",
                    'title': title or "Unknown Title"
                }
            
        except Exception as e:
            logger.debug(f"Could not extract paper info: {e}")
        
        return None
    
    def analyze_papers_with_nim(self, papers: List[Dict[str, str]], 
                               max_papers: Optional[int] = None,
                               batch_size: int = 5) -> List[PaperAnalysisResult]:
        """
        Analyze papers using NVIDIA NIM.
        
        Args:
            papers: List of paper dictionaries
            max_papers: Maximum number of papers to analyze (None for all)
            batch_size: Number of papers to process concurrently
            
        Returns:
            List of analysis results
        """
        # Limit papers if specified
        if max_papers:
            papers = papers[:max_papers]
        
        logger.info(f"Starting NIM analysis of {len(papers)} papers")
        
        # Test connection first
        if not self.analyzer.test_connection():
            logger.error("Cannot connect to NIM service. Please ensure it's running.")
            return []
        
        # Process papers in batches
        all_results = []
        total_batches = (len(papers) + batch_size - 1) // batch_size
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} papers)")
            
            try:
                results = self.analyzer.analyze_paper_batch(batch, max_concurrent=batch_size)
                all_results.extend(results)
                
                logger.info(f"Completed batch {batch_num}: {len(results)} successful analyses")
                
                # Small delay between batches to avoid overwhelming the service
                if i + batch_size < len(papers):
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                continue
        
        logger.info(f"NIM analysis completed: {len(all_results)} papers analyzed")
        return all_results
    
    def compare_with_bert_results(self, nim_results: List[PaperAnalysisResult], 
                                 bert_results_path: str = None) -> Dict[str, Any]:
        """
        Compare NIM results with BERT results.
        
        Args:
            nim_results: NIM analysis results
            bert_results_path: Path to BERT results file
            
        Returns:
            Comparison dictionary
        """
        if not bert_results_path:
            # Look for BERT results in common locations
            possible_paths = [
                'bert_training_output/pipeline_summary.json',
                'evaluation_results/evaluation_report.json',
                'bert_training_output/processed_data/evaluation_results.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    bert_results_path = path
                    break
        
        if not bert_results_path or not os.path.exists(bert_results_path):
            logger.warning("No BERT results found for comparison")
            return {'error': 'No BERT results available'}
        
        return self.analyzer.compare_with_bert_results(nim_results, bert_results_path)
    
    def generate_comprehensive_report(self, nim_results: List[PaperAnalysisResult],
                                    comparison_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        
        Args:
            nim_results: NIM analysis results
            comparison_data: Comparison with BERT results
            
        Returns:
            Comprehensive report dictionary
        """
        # Basic summary from NIM analyzer
        summary = self.analyzer.create_summary_report(nim_results)
        
        # Add detailed analysis
        report = {
            'analysis_summary': summary,
            'detailed_results': {
                'total_papers': len(nim_results),
                'successful_analyses': len([r for r in nim_results if r.confidence_score > 0.3]),
                'high_confidence_analyses': len([r for r in nim_results if r.confidence_score > 0.8]),
                'average_processing_time': sum(r.processing_time for r in nim_results) / len(nim_results) if nim_results else 0,
                'model_used': self.nim_config.model_name
            },
            'insights': {
                'most_confident_analysis': None,
                'fastest_analysis': None,
                'most_findings': None,
                'common_limitations': [],
                'research_trends': []
            }
        }
        
        if nim_results:
            # Find most confident analysis
            most_confident = max(nim_results, key=lambda x: x.confidence_score)
            report['insights']['most_confident_analysis'] = {
                'paper_id': most_confident.paper_id,
                'title': most_confident.title,
                'confidence_score': most_confident.confidence_score,
                'summary': most_confident.summary
            }
            
            # Find fastest analysis
            fastest = min(nim_results, key=lambda x: x.processing_time)
            report['insights']['fastest_analysis'] = {
                'paper_id': fastest.paper_id,
                'processing_time': fastest.processing_time
            }
            
            # Find paper with most findings
            most_findings = max(nim_results, key=lambda x: len(x.key_findings))
            report['insights']['most_findings'] = {
                'paper_id': most_findings.paper_id,
                'title': most_findings.title,
                'findings_count': len(most_findings.key_findings),
                'key_findings': most_findings.key_findings
            }
            
            # Analyze common limitations
            all_limitations = []
            for result in nim_results:
                all_limitations.extend(result.limitations)
            
            # Simple frequency analysis
            limitation_counts = {}
            for limitation in all_limitations:
                limitation_lower = limitation.lower()
                for key in limitation_counts:
                    if key in limitation_lower or limitation_lower in key:
                        limitation_counts[key] += 1
                        break
                else:
                    limitation_counts[limitation] = 1
            
            # Get most common limitations
            sorted_limitations = sorted(limitation_counts.items(), key=lambda x: x[1], reverse=True)
            report['insights']['common_limitations'] = sorted_limitations[:5]
        
        # Add comparison data if available
        if comparison_data:
            report['bert_comparison'] = comparison_data
        
        return report
    
    def save_results_and_report(self, nim_results: List[PaperAnalysisResult],
                               report: Dict[str, Any], 
                               prefix: str = "nim_analysis") -> Dict[str, str]:
        """
        Save analysis results and report to files.
        
        Args:
            nim_results: NIM analysis results
            report: Comprehensive report
            prefix: File prefix for output files
            
        Returns:
            Dictionary with paths to saved files
        """
        timestamp = int(time.time())
        
        # Save detailed results
        results_file = self.output_dir / f"{prefix}_results_{timestamp}.json"
        self.analyzer.save_results(nim_results, str(results_file))
        
        # Save comprehensive report
        report_file = self.output_dir / f"{prefix}_report_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save summary CSV for easy analysis
        csv_file = self.output_dir / f"{prefix}_summary_{timestamp}.csv"
        self._save_summary_csv(nim_results, str(csv_file))
        
        logger.info(f"Results saved to {self.output_dir}")
        
        return {
            'results_json': str(results_file),
            'report_json': str(report_file),
            'summary_csv': str(csv_file)
        }
    
    def _save_summary_csv(self, results: List[PaperAnalysisResult], csv_path: str):
        """Save a summary CSV file."""
        try:
            import pandas as pd
            
            data = []
            for result in results:
                data.append({
                    'paper_id': result.paper_id,
                    'title': result.title[:100] + '...' if len(result.title) > 100 else result.title,
                    'confidence_score': result.confidence_score,
                    'sentiment_score': result.sentiment_score,
                    'processing_time': result.processing_time,
                    'num_key_findings': len(result.key_findings),
                    'num_limitations': len(result.limitations),
                    'model_used': result.model_used
                })
            
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)
            logger.info(f"Summary CSV saved to {csv_path}")
            
        except ImportError:
            logger.warning("Pandas not available, skipping CSV export")
        except Exception as e:
            logger.error(f"Error saving CSV: {e}")


def main():
    """Main function for running the NIM paper pipeline."""
    parser = argparse.ArgumentParser(description='NVIDIA NIM Research Paper Analysis Pipeline')
    
    # Input options
    parser.add_argument('--data-dir', type=str, default='bert_training_output/processed_data',
                       help='Directory containing processed paper data')
    parser.add_argument('--data-file', type=str, 
                       help='Specific data file to analyze')
    
    # NIM options
    parser.add_argument('--model', type=str, default='llama-3.1-8b-instruct',
                       choices=['llama-3.1-8b-instruct', 'codellama-13b-instruct', 'mistral-7b-instruct'],
                       help='NIM model to use')
    parser.add_argument('--nim-url', type=str, default='http://localhost:8000/v1',
                       help='NIM service URL')
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='Maximum tokens for NIM responses')
    
    # Processing options
    parser.add_argument('--max-papers', type=int, 
                       help='Maximum number of papers to analyze')
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Batch size for concurrent processing')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='nim_analysis_output',
                       help='Output directory for results')
    parser.add_argument('--prefix', type=str, default='nim_analysis',
                       help='Prefix for output files')
    
    # Actions
    parser.add_argument('--deploy-nim', action='store_true',
                       help='Deploy NIM service before analysis')
    parser.add_argument('--check-service', action='store_true',
                       help='Only check if NIM service is running')
    
    args = parser.parse_args()
    
    print("🚀 NVIDIA NIM Research Paper Analysis Pipeline")
    print("=" * 60)
    
    # Create NIM configuration
    nim_config = NIMConfig(
        base_url=args.nim_url,
        model_name=args.model,
        max_tokens=args.max_tokens
    )
    
    # Initialize pipeline
    pipeline = PaperPipelineIntegration(nim_config, args.output_dir)
    
    # Deploy NIM service if requested
    if args.deploy_nim:
        print("\n🚀 Deploying NIM service...")
        deployment_manager = NIMDeploymentManager()
        if not deployment_manager.check_prerequisites():
            print("❌ Prerequisites not met. Cannot deploy NIM service.")
            return
        
        success = deployment_manager.deploy_model(args.model)
        if not success:
            print("❌ Failed to deploy NIM service")
            return
        
        print("✅ NIM service deployed successfully")
    
    # Check service connection
    if args.check_service:
        if pipeline.analyzer.test_connection():
            print("✅ NIM service is running and accessible")
        else:
            print("❌ Cannot connect to NIM service")
            print(f"Make sure NIM is running at {args.nim_url}")
            return
        return
    
    # Load papers
    print(f"\n📚 Loading papers...")
    if args.data_file:
        papers = pipeline.load_processed_papers(args.data_file)
    else:
        papers = pipeline.extract_paper_text_from_processed_data(args.data_dir)
    
    if not papers:
        print("❌ No papers found to analyze")
        return
    
    print(f"📊 Found {len(papers)} papers")
    if args.max_papers:
        papers = papers[:args.max_papers]
        print(f"📝 Limited to {len(papers)} papers for analysis")
    
    # Analyze papers with NIM
    print(f"\n🤖 Starting NIM analysis with {args.model}...")
    nim_results = pipeline.analyze_papers_with_nim(papers, args.max_papers, args.batch_size)
    
    if not nim_results:
        print("❌ No successful analyses completed")
        return
    
    # Compare with BERT results if available
    print("\n📊 Comparing with BERT results...")
    comparison_data = pipeline.compare_with_bert_results(nim_results)
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    report = pipeline.generate_comprehensive_report(nim_results, comparison_data)
    
    # Save results
    print("\n💾 Saving results...")
    saved_files = pipeline.save_results_and_report(nim_results, report, args.prefix)
    
    # Print summary
    print("\n✅ Analysis Complete!")
    print("=" * 40)
    print(f"📊 Papers analyzed: {len(nim_results)}")
    print(f"🎯 Average confidence: {report['analysis_summary']['average_confidence_score']:.3f}")
    print(f"⏱️  Average processing time: {report['analysis_summary']['average_processing_time_seconds']:.2f}s")
    print(f"🤖 Model used: {args.model}")
    
    print("\n📁 Output files:")
    for file_type, file_path in saved_files.items():
        print(f"   {file_type}: {file_path}")
    
    print(f"\n🎉 Analysis completed successfully!")


if __name__ == "__main__":
    main()
