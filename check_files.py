from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_data_files():
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
        logger.info(f"  {file.name} ({file.stat().st_size / 1024:.2f} KB)")

def cleanup_intermediate_files():
    data_dir = Path('data')
    if not data_dir.exists():
        logger.error(f"Data directory not found at {data_dir.absolute()}")
        return
        
    # Find all intermediate files
    intermediate_files = list(data_dir.glob('pubmed_ethics_papers_intermediate_*.json'))
    year_files = list(data_dir.glob('pubmed_ethics_papers_20*.json'))
    
    total_files = len(intermediate_files) + len(year_files)
    if total_files == 0:
        logger.info("No intermediate files found to clean up")
        return
        
    logger.info(f"Found {total_files} files to clean up")
    
    # Remove files
    for file in intermediate_files + year_files:
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
    logger.info("Current files:")
    list_data_files()
    
    response = input("\nDo you want to remove intermediate files? (y/n): ")
    if response.lower() == 'y':
        cleanup_intermediate_files() 