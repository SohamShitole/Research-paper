#!/usr/bin/env python3
"""
Script to download research papers from the SciTLDR dataset.

This script downloads papers using OpenReview API and multiple approaches:
1. OpenReview API to get paper metadata and PDF links
2. Direct arXiv download if arXiv ID is available
3. Semantic Scholar API as fallback
4. DOI resolution if available

Features:
- Skip already downloaded papers (checks both log files and actual PDF existence)
- Separate folders for different CSV files (train/test/val)
- Force re-download option
- Comprehensive error handling and logging
- Rate limiting and resume capability

Usage:
    python download_papers.py

Requirements:
    pip install pandas requests tqdm
"""

import os
import sys
import time
import pandas as pd
import requests
from typing import Set, Dict, List, Optional
from pathlib import Path
import json
from tqdm import tqdm
import argparse

class PaperDownloader:
    def __init__(self, output_dir: str = "downloaded_papers", delay: float = 1.0, force_redownload: bool = False):
        """
        Initialize the paper downloader.
        
        Args:
            output_dir: Directory to save downloaded papers
            delay: Delay between API requests to avoid rate limiting
            force_redownload: If True, ignore previous downloads and re-download everything
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.delay = delay
        self.force_redownload = force_redownload
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SciTLDR-Paper-Downloader/1.0 (Educational Research)'
        })
        
        # Statistics
        self.stats = {
            'total_papers': 0,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0,
            'openreview_found': 0,
            'semantic_scholar_found': 0,
            'arxiv_found': 0,
            'doi_found': 0
        }
        
        # Create subdirectories
        (self.output_dir / "pdfs").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "failed").mkdir(exist_ok=True)
        
        # Load failed downloads to skip them
        self.failed_ids_file = self.output_dir / "failed" / "failed_downloads.txt"
        self.failed_ids = set() if force_redownload else self.load_failed_ids()
        
        # Load successful downloads to skip them
        self.success_log = self.output_dir / "successful_downloads.txt"
        self.successful_ids = set() if force_redownload else self.load_successful_ids()

    def load_failed_ids(self) -> Set[str]:
        """Load previously failed download IDs."""
        if self.failed_ids_file.exists():
            with open(self.failed_ids_file, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def load_successful_ids(self) -> Set[str]:
        """Load previously successful download IDs."""
        if self.success_log.exists():
            with open(self.success_log, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def log_failed_download(self, paper_id: str, error: str):
        """Log a failed download."""
        with open(self.failed_ids_file, 'a') as f:
            f.write(f"{paper_id}\n")
        
        error_log = self.output_dir / "failed" / f"{paper_id}_error.txt"
        with open(error_log, 'w') as f:
            f.write(f"Paper ID: {paper_id}\n")
            f.write(f"Error: {error}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def log_successful_download(self, paper_id: str):
        """Log a successful download."""
        with open(self.success_log, 'a') as f:
            f.write(f"{paper_id}\n")

    def get_openreview_metadata(self, paper_id: str) -> Optional[Dict]:
        """
        Get paper metadata from OpenReview API.
        
        Args:
            paper_id: The paper ID from SciTLDR dataset (OpenReview ID)
            
        Returns:
            Dictionary with paper metadata or None if not found
        """
        try:
            url = f"https://api.openreview.net/notes?id={paper_id}"
            
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('notes') and len(data['notes']) > 0:
                    self.stats['openreview_found'] += 1
                    note = data['notes'][0]
                    
                    # Extract relevant information
                    content = note.get('content', {})
                    metadata = {
                        'id': paper_id,
                        'title': content.get('title', 'N/A'),
                        'abstract': content.get('abstract', 'N/A'),
                        'authors': content.get('authors', []),
                        'venue': note.get('forum', 'N/A'),
                        'year': note.get('cdate', 0) // 1000 // 31536000 + 1970 if note.get('cdate') else None,
                        'pdf_url': None,  # Will be filled if available
                        'openreview_url': f"https://openreview.net/forum?id={paper_id}",
                        'source': 'openreview'
                    }
                    
                    # Check for PDF attachment
                    if note.get('content', {}).get('pdf'):
                        # Some papers have direct PDF links in content
                        pdf_link = note['content']['pdf']
                        if isinstance(pdf_link, str) and pdf_link.startswith('http'):
                            metadata['pdf_url'] = pdf_link
                    
                    return metadata
                else:
                    return None
            elif response.status_code == 429:
                print(f"Rate limited by OpenReview API. Waiting longer...")
                time.sleep(10)  # Wait longer for rate limit
                return None
            else:
                return None
                
        except Exception as e:
            print(f"Error fetching OpenReview metadata for {paper_id}: {e}")
            return None

    def get_semantic_scholar_metadata(self, paper_id: str) -> Optional[Dict]:
        """
        Get paper metadata from Semantic Scholar API (fallback).
        
        Args:
            paper_id: The paper ID from SciTLDR dataset
            
        Returns:
            Dictionary with paper metadata or None if not found
        """
        try:
            url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
            params = {
                'fields': 'title,authors,year,externalIds,url,openAccessPdf'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                self.stats['semantic_scholar_found'] += 1
                return response.json()
            elif response.status_code == 429:
                print(f"Rate limited by Semantic Scholar API. Waiting longer...")
                time.sleep(10)  # Wait longer for rate limit
                return None
            elif response.status_code == 404:
                return None
            else:
                print(f"Semantic Scholar API error {response.status_code} for {paper_id}")
                return None
                
        except Exception as e:
            print(f"Error fetching metadata for {paper_id}: {e}")
            return None

    def download_from_url(self, url: str, filename: str) -> bool:
        """
        Download a PDF from a given URL.
        
        Args:
            url: URL to download from
            filename: Local filename to save to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.session.get(url, timeout=60, stream=True)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and not url.endswith('.pdf'):
                print(f"Warning: Downloaded file might not be a PDF (Content-Type: {content_type})")
            
            filepath = self.output_dir / "pdfs" / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify the file is not empty
            if filepath.stat().st_size == 0:
                filepath.unlink()  # Remove empty file
                return False
            
            return True
            
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
            return False

    def try_arxiv_download(self, arxiv_id: str, paper_id: str) -> bool:
        """Try to download paper from arXiv."""
        if not arxiv_id:
            return False
            
        # Clean arXiv ID (remove version if present)
        clean_arxiv_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id
        
        url = f"https://arxiv.org/pdf/{clean_arxiv_id}.pdf"
        filename = f"{paper_id}_arxiv_{clean_arxiv_id}.pdf"
        
        if self.download_from_url(url, filename):
            self.stats['arxiv_found'] += 1
            return True
        return False

    def try_openreview_pdf(self, paper_id: str) -> bool:
        """Try to download PDF directly from OpenReview."""
        # OpenReview PDF URLs typically follow this pattern
        pdf_url = f"https://openreview.net/pdf?id={paper_id}"
        filename = f"{paper_id}_openreview.pdf"
        
        return self.download_from_url(pdf_url, filename)

    def try_open_access_pdf(self, open_access_pdf: Dict, paper_id: str) -> bool:
        """Try to download from open access PDF URL."""
        if not open_access_pdf or 'url' not in open_access_pdf:
            return False
            
        url = open_access_pdf['url']
        filename = f"{paper_id}_open_access.pdf"
        
        return self.download_from_url(url, filename)

    def try_doi_resolution(self, doi: str, paper_id: str) -> bool:
        """Try to resolve DOI and find open access version."""
        if not doi:
            return False
            
        # This is a basic implementation - in practice, DOI resolution
        # for open access papers is complex and would need more sophisticated logic
        return False

    def save_metadata(self, paper_id: str, metadata: Dict):
        """Save paper metadata to JSON file."""
        metadata_file = self.output_dir / "metadata" / f"{paper_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def download_paper(self, paper_id: str) -> bool:
        """
        Download a single paper by its ID.
        
        Args:
            paper_id: The paper ID from SciTLDR dataset
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if already downloaded or failed (unless force redownload)
        if not self.force_redownload and paper_id in self.successful_ids:
            # Double-check that the PDF file actually exists
            pdf_files = list((self.output_dir / "pdfs").glob(f"{paper_id}_*.pdf"))
            if pdf_files and pdf_files[0].exists() and pdf_files[0].stat().st_size > 0:
                self.stats['skipped'] += 1
                return True
            else:
                # File is missing or empty, remove from successful list and re-download
                print(f"  PDF file missing or empty for {paper_id}, re-downloading...")
                self.successful_ids.discard(paper_id)
        
        if not self.force_redownload and paper_id in self.failed_ids:
            self.stats['skipped'] += 1
            return False
        
        print(f"Processing paper: {paper_id}")
        
        try:
            # Try OpenReview first (most likely source for SciTLDR papers)
            metadata = self.get_openreview_metadata(paper_id)
            
            if metadata:
                # Save metadata
                self.save_metadata(paper_id, metadata)
                
                # Try OpenReview direct PDF download first
                if self.try_openreview_pdf(paper_id):
                    self.log_successful_download(paper_id)
                    self.stats['downloaded'] += 1
                    return True
                
                # If OpenReview PDF didn't work, try other sources from metadata
                if metadata.get('pdf_url'):
                    filename = f"{paper_id}_direct.pdf"
                    if self.download_from_url(metadata['pdf_url'], filename):
                        self.log_successful_download(paper_id)
                        self.stats['downloaded'] += 1
                        return True
            
            # Fallback to Semantic Scholar if OpenReview didn't work
            semantic_metadata = self.get_semantic_scholar_metadata(paper_id)
            
            if semantic_metadata:
                # Save semantic scholar metadata
                self.save_metadata(paper_id + "_semantic", semantic_metadata)
                
                # Try different download sources from Semantic Scholar
                external_ids = semantic_metadata.get('externalIds', {})
                open_access_pdf = semantic_metadata.get('openAccessPdf')
                
                # Try arXiv (usually most reliable)
                if 'ArXiv' in external_ids:
                    if self.try_arxiv_download(external_ids['ArXiv'], paper_id):
                        self.log_successful_download(paper_id)
                        self.stats['downloaded'] += 1
                        return True
                
                # Try open access PDF
                if self.try_open_access_pdf(open_access_pdf, paper_id):
                    self.log_successful_download(paper_id)
                    self.stats['downloaded'] += 1
                    return True
                
                # Try DOI resolution (basic implementation)
                if 'DOI' in external_ids:
                    if self.try_doi_resolution(external_ids['DOI'], paper_id):
                        self.log_successful_download(paper_id)
                        self.stats['downloaded'] += 1
                        return True
            
            # If we have metadata but no download succeeded
            if metadata or semantic_metadata:
                error_msg = f"Found metadata but no downloadable PDF available"
                print(f"  {error_msg}")
                self.log_failed_download(paper_id, error_msg)
                self.stats['failed'] += 1
                return False
            else:
                # No metadata found anywhere
                error_msg = "Paper not found in OpenReview or Semantic Scholar"
                print(f"  {error_msg}")
                self.log_failed_download(paper_id, error_msg)
                self.stats['failed'] += 1
                return False
        
        except KeyboardInterrupt:
            print(f"\nDownload interrupted by user for {paper_id}")
            raise
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"  {error_msg}")
            self.log_failed_download(paper_id, error_msg)
            self.stats['failed'] += 1
            return False

    def download_from_csv(self, csv_file: str, separate_folders: bool = False):
        """Download papers from a CSV file."""
        print(f"Loading papers from {csv_file}")
        df = pd.read_csv(csv_file)
        paper_ids = df['paper_id'].unique().tolist()
        
        print(f"Found {len(paper_ids)} unique papers in {csv_file}")
        
        # Create separate subdirectories if requested
        if separate_folders:
            csv_name = Path(csv_file).stem  # e.g., 'train', 'test', 'val'
            csv_output_dir = self.output_dir / csv_name
            csv_output_dir.mkdir(exist_ok=True)
            (csv_output_dir / "pdfs").mkdir(exist_ok=True)
            (csv_output_dir / "metadata").mkdir(exist_ok=True)
            (csv_output_dir / "failed").mkdir(exist_ok=True)
            
            # Temporarily change output directories
            original_output_dir = self.output_dir
            original_failed_ids_file = self.failed_ids_file
            original_success_log = self.success_log
            
            self.output_dir = csv_output_dir
            self.failed_ids_file = csv_output_dir / "failed" / "failed_downloads.txt"
            self.success_log = csv_output_dir / "successful_downloads.txt"
            
            # Reload failed and successful IDs for this specific dataset
            if not self.force_redownload:
                self.failed_ids = self.load_failed_ids()
                self.successful_ids = self.load_successful_ids()
            else:
                self.failed_ids = set()
                self.successful_ids = set()
        
        try:
            for paper_id in tqdm(paper_ids, desc=f"Downloading from {csv_file}"):
                try:
                    self.download_paper(paper_id)
                    time.sleep(self.delay)  # Rate limiting
                except KeyboardInterrupt:
                    print(f"\nDownload interrupted by user. Progress saved.")
                    break
                except Exception as e:
                    print(f"Unexpected error processing {paper_id}: {e}")
                    continue
        
        finally:
            # Restore original directories if they were changed
            if separate_folders:
                self.output_dir = original_output_dir
                self.failed_ids_file = original_failed_ids_file
                self.success_log = original_success_log
        
        return paper_ids

    def download_all_datasets(self, csv_files: List[str], separate_folders: bool = False):
        """Download papers from all CSV files."""
        if separate_folders:
            # Download from each CSV separately into different folders
            total_papers = 0
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    print(f"\n{'='*50}")
                    print(f"Processing {csv_file}")
                    print('='*50)
                    try:
                        paper_ids = self.download_from_csv(csv_file, separate_folders=True)
                        total_papers += len(paper_ids)
                    except KeyboardInterrupt:
                        print(f"\nDownload interrupted by user.")
                        break
                    except Exception as e:
                        print(f"Error processing {csv_file}: {e}")
                        continue
                else:
                    print(f"Warning: {csv_file} not found, skipping.")
            
            self.stats['total_papers'] = total_papers
        else:
            # Original behavior: combine all papers into one folder
            all_paper_ids = set()
            
            # Collect all unique paper IDs first
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    df = pd.read_csv(csv_file)
                    all_paper_ids.update(df['paper_id'].unique())
                else:
                    print(f"Warning: {csv_file} not found, skipping.")
            
            print(f"Total unique papers across all datasets: {len(all_paper_ids)}")
            self.stats['total_papers'] = len(all_paper_ids)
            
            # Download each paper
            try:
                for paper_id in tqdm(list(all_paper_ids), desc="Downloading papers"):
                    try:
                        self.download_paper(paper_id)
                        time.sleep(self.delay)  # Rate limiting
                    except KeyboardInterrupt:
                        print(f"\nDownload interrupted by user. Progress saved.")
                        break
                    except Exception as e:
                        print(f"Unexpected error processing {paper_id}: {e}")
                        continue
            except KeyboardInterrupt:
                print(f"\nDownload process interrupted by user.")

    def print_statistics(self):
        """Print download statistics."""
        print("\n" + "="*50)
        print("DOWNLOAD STATISTICS")
        print("="*50)
        print(f"Total papers: {self.stats['total_papers']}")
        print(f"Successfully downloaded: {self.stats['downloaded']}")
        print(f"Failed downloads: {self.stats['failed']}")
        print(f"Skipped (already done): {self.stats['skipped']}")
        print(f"Found via OpenReview: {self.stats['openreview_found']}")
        print(f"Found via Semantic Scholar: {self.stats['semantic_scholar_found']}")
        print(f"Downloaded from arXiv: {self.stats['arxiv_found']}")
        print(f"Success rate: {self.stats['downloaded']/(self.stats['total_papers'])*100:.1f}%" 
              if self.stats['total_papers'] > 0 else "N/A")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Download papers from SciTLDR dataset')
    parser.add_argument('--output-dir', default='downloaded_papers', 
                       help='Directory to save downloaded papers')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between requests in seconds')
    parser.add_argument('--csv-files', nargs='+', 
                       default=['train.csv', 'test.csv', 'val.csv'],
                       help='CSV files to process')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: only process first 10 papers')
    parser.add_argument('--separate-folders', action='store_true',
                       help='Create separate folders for each CSV file (train/test/val)')
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force re-download of all papers, ignoring previous downloads')
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = PaperDownloader(output_dir=args.output_dir, delay=args.delay, force_redownload=args.force_redownload)
    
    print("SciTLDR Paper Downloader")
    print("="*30)
    print(f"Output directory: {args.output_dir}")
    print(f"Request delay: {args.delay} seconds")
    print(f"CSV files: {args.csv_files}")
    print(f"Separate folders: {args.separate_folders}")
    print(f"Force re-download: {args.force_redownload}")
    
    try:
        if args.test_mode:
            print("Running in TEST MODE (first 10 papers only)")
            # Test with just a few papers
            if os.path.exists(args.csv_files[0]):
                df = pd.read_csv(args.csv_files[0])
                test_ids = df['paper_id'].head(10).tolist()
                downloader.stats['total_papers'] = len(test_ids)
                
                for paper_id in tqdm(test_ids, desc="Test downloading"):
                    try:
                        downloader.download_paper(paper_id)
                        time.sleep(args.delay)
                    except KeyboardInterrupt:
                        print(f"\nTest interrupted by user.")
                        break
                    except Exception as e:
                        print(f"Error in test mode: {e}")
                        continue
            else:
                print(f"Error: {args.csv_files[0]} not found")
                return
        else:
            # Download from all datasets
            try:
                downloader.download_all_datasets(args.csv_files, separate_folders=args.separate_folders)
            except KeyboardInterrupt:
                print(f"\nDownload process interrupted by user.")
            except Exception as e:
                print(f"Unexpected error: {e}")
    
    except KeyboardInterrupt:
        print(f"\nProgram interrupted by user. Progress has been saved.")
    
    # Print final statistics
    downloader.print_statistics()
    
    if args.separate_folders:
        print(f"\nDownloaded papers saved to:")
        print(f"  - Train: {args.output_dir}/train/pdfs/")
        print(f"  - Test: {args.output_dir}/test/pdfs/")
        print(f"  - Validation: {args.output_dir}/val/pdfs/")
        print(f"Metadata and logs saved in respective folders")
    else:
        print(f"\nDownloaded papers saved to: {args.output_dir}/pdfs/")
        print(f"Metadata saved to: {args.output_dir}/metadata/")
        print(f"Failed downloads logged in: {args.output_dir}/failed/")

if __name__ == "__main__":
    main()