#!/usr/bin/env python3
"""
Data preprocessing pipeline for research papers.

This module handles:
1. PDF text extraction from downloaded papers
2. Text cleaning and preprocessing 
3. Dataset creation for BERT training
4. Data splitting and tokenization
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# PDF processing
import PyPDF2
import pdfplumber

# Text processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# ML utilities
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extract text from PDF files using multiple methods."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (better for complex layouts)."""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text using the best available method."""
        # Try pdfplumber first (generally better quality)
        text = self.extract_with_pdfplumber(pdf_path)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text.strip():
            text = self.extract_with_pypdf2(pdf_path)
        
        return text.strip()


class TextCleaner:
    """Clean and preprocess extracted text."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Remove lines that are too short (likely headers/footers)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from the paper."""
        sections = {
            'abstract': '',
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': '',
            'full_text': text
        }
        
        # Simple heuristic-based section extraction
        text_lower = text.lower()
        
        # Extract abstract
        abstract_patterns = [
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*(?:1\.|introduction|keywords))',
            r'abstract\s*[:\-]?\s*(.*?)(?=\n\s*\n)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                sections['abstract'] = match.group(1).strip()
                break
        
        # Extract introduction
        intro_match = re.search(
            r'(?:1\s*\.?\s*)?introduction\s*[:\-]?\s*(.*?)(?=\n\s*(?:2\.|methodology|method|approach|related work))',
            text_lower, re.DOTALL | re.IGNORECASE
        )
        if intro_match:
            sections['introduction'] = intro_match.group(1).strip()
        
        return sections


class PaperDatasetCreator:
    """Create dataset from processed papers."""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.extractor = PDFTextExtractor()
        self.cleaner = TextCleaner()
    
    def process_papers(self, split: str = 'train') -> List[Dict]:
        """Process papers from a specific split."""
        split_dir = self.base_dir / split
        pdf_dir = split_dir / 'pdfs'
        metadata_dir = split_dir / 'metadata'
        
        if not pdf_dir.exists():
            logger.error(f"PDF directory not found: {pdf_dir}")
            return []
        
        papers = []
        pdf_files = list(pdf_dir.glob('*.pdf'))
        
        logger.info(f"Processing {len(pdf_files)} papers from {split} split")
        
        for pdf_file in tqdm(pdf_files, desc=f"Processing {split} papers"):
            try:
                # Extract paper ID from filename
                paper_id = pdf_file.stem.split('_')[0]
                
                # Load metadata if available
                metadata_file = metadata_dir / f"{paper_id}_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Extract text from PDF
                text = self.extractor.extract_text(str(pdf_file))
                if not text:
                    logger.warning(f"No text extracted from {pdf_file}")
                    continue
                
                # Clean text
                cleaned_text = self.cleaner.clean_text(text)
                if not cleaned_text:
                    logger.warning(f"No text after cleaning {pdf_file}")
                    continue
                
                # Extract sections
                sections = self.cleaner.extract_sections(cleaned_text)
                
                # Create paper record
                paper_record = {
                    'paper_id': paper_id,
                    'title': metadata.get('title', ''),
                    'abstract': metadata.get('abstract', sections['abstract']),
                    'authors': metadata.get('authors', []),
                    'year': metadata.get('year', None),
                    'venue': metadata.get('venue', ''),
                    'full_text': sections['full_text'],
                    'introduction': sections['introduction'],
                    'word_count': len(cleaned_text.split()),
                    'char_count': len(cleaned_text),
                    'split': split
                }
                
                papers.append(paper_record)
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(papers)} papers from {split} split")
        return papers
    
    def create_bert_dataset(self, papers: List[Dict], max_length: int = 512) -> List[Dict]:
        """Create BERT training dataset from papers."""
        bert_samples = []
        
        for paper in tqdm(papers, desc="Creating BERT samples"):
            text = paper['full_text']
            
            # Split text into sentences
            sentences = sent_tokenize(text)
            
            # Create samples with different strategies
            # Strategy 1: Abstract + Introduction pairs
            if paper['abstract'] and paper['introduction']:
                bert_samples.append({
                    'text_a': paper['abstract'][:max_length//2],
                    'text_b': paper['introduction'][:max_length//2],
                    'label': 1,  # Related texts
                    'paper_id': paper['paper_id'],
                    'sample_type': 'abstract_intro'
                })
            
            # Strategy 2: Title + Abstract pairs
            if paper['title'] and paper['abstract']:
                bert_samples.append({
                    'text_a': paper['title'],
                    'text_b': paper['abstract'][:max_length-len(paper['title'])],
                    'label': 1,  # Related texts
                    'paper_id': paper['paper_id'],
                    'sample_type': 'title_abstract'
                })
            
            # Strategy 3: Consecutive sentence pairs (Next Sentence Prediction)
            for i in range(len(sentences) - 1):
                if len(sentences[i]) > 20 and len(sentences[i+1]) > 20:
                    bert_samples.append({
                        'text_a': sentences[i][:max_length//2],
                        'text_b': sentences[i+1][:max_length//2],
                        'label': 1,  # Next sentence
                        'paper_id': paper['paper_id'],
                        'sample_type': 'next_sentence'
                    })
                    
                    # Create negative samples (random sentence pairs)
                    if np.random.random() < 0.3:  # 30% negative samples
                        random_idx = np.random.randint(0, len(sentences))
                        if random_idx != i and random_idx != i+1:
                            bert_samples.append({
                                'text_a': sentences[i][:max_length//2],
                                'text_b': sentences[random_idx][:max_length//2],
                                'label': 0,  # Not next sentence
                                'paper_id': paper['paper_id'],
                                'sample_type': 'random_sentence'
                            })
        
        logger.info(f"Created {len(bert_samples)} BERT training samples")
        return bert_samples
    
    def save_dataset(self, papers: List[Dict], output_path: str):
        """Save processed dataset."""
        df = pd.DataFrame(papers)
        df.to_json(output_path, orient='records', indent=2)
        logger.info(f"Saved dataset to {output_path}")
    
    def save_bert_dataset(self, bert_samples: List[Dict], output_path: str):
        """Save BERT training dataset."""
        df = pd.DataFrame(bert_samples)
        df.to_json(output_path, orient='records', indent=2)
        logger.info(f"Saved BERT dataset to {output_path}")


class BERTDatasetTokenizer:
    """Tokenize dataset for BERT training."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False)
        except Exception as e:
            logger.error(f"Failed to load tokenizer for {model_name}: {e}")
            logger.info("Trying alternative approaches...")
            
            # Try alternative models
            alternatives = ['distilbert-base-uncased', 'bert-base-cased']
            for alt_model in alternatives:
                try:
                    logger.info(f"Trying alternative model: {alt_model}")
                    self.tokenizer = AutoTokenizer.from_pretrained(alt_model, use_auth_token=False)
                    logger.info(f"Successfully loaded {alt_model}")
                    break
                except Exception:
                    continue
            else:
                # If all fails, provide helpful error message
                error_msg = f"""
❌ Failed to load any tokenizer. This might be due to:
1. Hugging Face authentication issues
2. Network connectivity problems
3. Model access restrictions

🔧 Solutions:
1. Run: python3 setup_huggingface.py
2. Or set up Hugging Face authentication:
   - Visit: https://huggingface.co/settings/tokens
   - Create a token and run: huggingface-cli login
3. Or use offline mode if models are cached

Original error: {e}
"""
                raise RuntimeError(error_msg)
    
    def tokenize_dataset(self, dataset_path: str, output_dir: str, max_length: int = 512):
        """Tokenize BERT dataset."""
        # Load dataset
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Tokenize
        tokenized_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
            encoding = self.tokenizer(
                row['text_a'],
                row['text_b'] if 'text_b' in row else None,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            
            tokenized_sample = {
                'input_ids': encoding['input_ids'].squeeze().tolist(),
                'attention_mask': encoding['attention_mask'].squeeze().tolist(),
                'token_type_ids': encoding['token_type_ids'].squeeze().tolist(),
                'labels': row['label'] if 'label' in row else 0,
                'paper_id': row['paper_id'],
                'sample_type': row.get('sample_type', 'unknown')
            }
            
            tokenized_data.append(tokenized_sample)
        
        # Save tokenized dataset
        output_path = Path(output_dir) / 'tokenized_dataset.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
        with open(output_path, 'w') as f:
            json.dump(tokenized_data, f, indent=2)
        
        logger.info(f"Tokenized dataset saved to {output_path}")
        return str(output_path)


def main():
    """Main preprocessing pipeline."""
    base_dir = "downloaded_papers"
    output_dir = "processed_data"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize dataset creator
    creator = PaperDatasetCreator(base_dir)
    
    # Process training papers
    train_papers = creator.process_papers('train')
    if train_papers:
        creator.save_dataset(train_papers, f"{output_dir}/train_papers.json")
        
        # Create BERT training dataset
        bert_dataset = creator.create_bert_dataset(train_papers)
        creator.save_bert_dataset(bert_dataset, f"{output_dir}/bert_train_dataset.json")
        
        # Tokenize dataset
        tokenizer = BERTDatasetTokenizer()
        tokenizer.tokenize_dataset(
            f"{output_dir}/bert_train_dataset.json",
            output_dir
        )
    
    # Process validation papers
    val_papers = creator.process_papers('val')
    if val_papers:
        creator.save_dataset(val_papers, f"{output_dir}/val_papers.json")
        
        # Create validation BERT dataset
        val_bert_dataset = creator.create_bert_dataset(val_papers)
        creator.save_bert_dataset(val_bert_dataset, f"{output_dir}/bert_val_dataset.json")
    
    # Process test papers
    test_papers = creator.process_papers('test')
    if test_papers:
        creator.save_dataset(test_papers, f"{output_dir}/test_papers.json")
    
    logger.info("Preprocessing pipeline completed!")


if __name__ == "__main__":
    main()

