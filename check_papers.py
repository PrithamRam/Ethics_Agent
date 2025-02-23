import json
from pathlib import Path
import glob
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_paper_file(file_path, description=""):
    with open(file_path) as f:
        papers = json.load(f)
        logger.info(f"\n{description} ({file_path.name}) contains {len(papers)} papers")
        
        # Count papers per year
        year_counts = Counter(paper.get('year', 'unknown') for paper in papers)
        
        # Check for duplicates
        pmids = [p.get('pubmed_id') for p in papers]
        unique_pmids = set(pmids)
        if len(pmids) != len(unique_pmids):
            logger.warning(f"Found {len(pmids) - len(unique_pmids)} duplicate papers!")
        
        logger.info("Year distribution:")
        for year in sorted(year_counts.keys()):
            logger.info(f"  {year}: {year_counts[year]} papers")
            
        return papers, year_counts, len(unique_pmids)

def check_papers():
    data_dir = Path('data')
    
    if not data_dir.exists():
        logger.error("Data directory not found!")
        return
        
    # Check error recovery file
    error_file = data_dir / 'pubmed_ethics_papers_error_recovery.json'
    if error_file.exists():
        error_papers, error_counts, unique_count = analyze_paper_file(error_file, "Error recovery file")
        logger.info(f"Number of unique papers in error recovery: {unique_count}")

    # Check all intermediate files
    intermediate_files = list(data_dir.glob('pubmed_ethics_papers_intermediate_*.json'))
    if intermediate_files:
        latest_file = max(intermediate_files, key=lambda x: int(x.stem.split('_')[-1]))
        analyze_paper_file(latest_file, "Latest intermediate file")

    # Check year-specific files
    year_files = list(data_dir.glob('pubmed_ethics_papers_20*.json'))
    if year_files:
        for file in sorted(year_files):
            analyze_paper_file(file, f"Year file")
    
    # Check main output file
    main_file = data_dir / 'pubmed_ethics_papers.json'
    if main_file.exists():
        main_papers, main_counts, unique_count = analyze_paper_file(main_file, "Main output file")
        logger.info(f"\nNumber of unique papers in main file: {unique_count}")
        
        # Analyze a sample of 2025 papers
        if '2025' in main_counts:
            logger.info("\nAnalyzing sample of 2025 papers:")
            future_papers = [p for p in main_papers if p.get('year') == '2025']
            for i, paper in enumerate(future_papers[:5]):
                logger.info(f"\nPaper {i+1}:")
                logger.info(f"Title: {paper.get('title')}")
                logger.info(f"Journal: {paper.get('journal')}")
                logger.info(f"PubMed ID: {paper.get('pubmed_id')}")

if __name__ == "__main__":
    check_papers() 