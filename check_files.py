from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_data_files():
    """Check if required data files exist and are valid"""
    data_dir = Path('data')
    
    # Check data directory
    if not data_dir.exists():
        logger.warning("Creating data directory")
        data_dir.mkdir(parents=True)
    
    # Check papers file
    papers_file = data_dir / 'pubmed_papers.json'
    if not papers_file.exists():
        logger.error(f"Missing papers file: {papers_file}")
        create_sample_papers(papers_file)
    else:
        # Validate JSON format
        try:
            with open(papers_file) as f:
                papers = json.load(f)
                logger.info(f"Found {len(papers)} papers in {papers_file}")
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {papers_file}")
            create_sample_papers(papers_file)

def create_sample_papers(file_path):
    """Create a sample papers file"""
    sample_papers = [
        {
            "pubmed_id": "12345678",
            "title": "Cultural Competence in Healthcare: A Systematic Review",
            "authors": [
                {"first_name": "John", "last_name": "Smith"},
                {"first_name": "Jane", "last_name": "Doe"}
            ],
            "abstract": "This systematic review examines cultural competence in healthcare settings...",
            "journal": "Journal of Medical Ethics",
            "year": 2020,
            "keywords": ["cultural competence", "healthcare", "ethics"],
            "relevance_score": 10
        },
        # Add more sample papers...
    ]
    
    with open(file_path, 'w') as f:
        json.dump(sample_papers, f, indent=2)
    logger.info(f"Created sample papers file at {file_path}")

def list_data_files():
    """List all data files and their contents"""
    data_dir = Path('data')
    if not data_dir.exists():
        logger.error(f"Data directory not found at {data_dir.absolute()}")
        return
        
    logger.info(f"Checking data directory: {data_dir.absolute()}")
    json_files = list(data_dir.glob('*.json'))
    
    if not json_files:
        logger.info("No JSON files found")
        return
        
    logger.info(f"Found {len(json_files)} JSON files:")
    for file in sorted(json_files):
        try:
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    logger.info(f"  {file.name}: {len(data)} papers")
                    # Show sample of papers
                    if len(data) > 0:
                        logger.info("  Sample papers:")
                        for paper in data[:3]:
                            logger.info(f"    - {paper.get('title', 'No title')} ({paper.get('year', 'No year')})")
                else:
                    logger.info(f"  {file.name}: Not a list of papers")
        except json.JSONDecodeError:
            logger.error(f"  {file.name}: Invalid JSON")
        except Exception as e:
            logger.error(f"  Error reading {file.name}: {e}")
        logger.info("")

def cleanup_intermediate_files():
    """Clean up intermediate and duplicate files, keeping only the main papers file"""
    data_dir = Path('data')
    if not data_dir.exists():
        logger.error(f"Data directory not found at {data_dir.absolute()}")
        return
        
    # First identify the best file to keep (most papers)
    best_file = None
    best_count = 0
    
    for file in data_dir.glob('*.json'):
        try:
            with open(file) as f:
                papers = json.load(f)
                if isinstance(papers, list):
                    if len(papers) > best_count:
                        best_count = len(papers)
                        best_file = file
        except Exception:
            continue
    
    if not best_file:
        logger.error("No valid papers file found")
        return
        
    # Rename best file to pubmed_papers.json if needed
    target_file = data_dir / 'pubmed_papers.json'
    if best_file != target_file:
        best_file.rename(target_file)
        logger.info(f"Renamed {best_file.name} to pubmed_papers.json")
    
    # Remove all other files
    files_to_remove = [f for f in data_dir.glob('*.json') if f != target_file]
    
    if not files_to_remove:
        logger.info("No files to clean up")
        return
        
    logger.info(f"Found {len(files_to_remove)} files to clean up:")
    for file in files_to_remove:
        logger.info(f"  {file.name}")
    
    response = input("\nProceed with cleanup? (y/n): ")
    if response.lower() != 'y':
        logger.info("Cleanup cancelled")
        return
        
    # Remove files
    for file in files_to_remove:
        try:
            file.unlink()
            logger.info(f"Removed {file.name}")
        except Exception as e:
            logger.error(f"Error removing {file.name}: {e}")
    
    logger.info("Cleanup complete")
    
    # Show remaining files
    logger.info("\nRemaining files:")
    list_data_files()

if __name__ == "__main__":
    logger.info("\nChecking data files...")
    list_data_files()
    
    response = input("\nDo you want to remove intermediate files? (y/n): ")
    if response.lower() == 'y':
        cleanup_intermediate_files() 