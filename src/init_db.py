import asyncio
import logging
from src.medical_ethics_assistant import MedicalEthicsAssistant
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def init_database():
    """Initialize and populate the database"""
    try:
        logger.info("Starting database initialization...")
        
        # Create assistant instance
        assistant = await MedicalEthicsAssistant.create()
        
        # Initialize database schema
        await assistant.ethics_db.initialize_database()
        logger.info("Database schema created")
        
        # Load data from JSON files
        await assistant.ethics_db.load_initial_data()
        
        # Clean up papers with insufficient ethical considerations
        logger.info("Cleaning up papers with insufficient ethical considerations...")
        await assistant.ethics_db.cleanup_database(min_considerations=2)
        
        # Verify final database contents
        if await assistant.ethics_db.verify_database():
            logger.info("Database initialized, cleaned, and verified successfully")
        else:
            logger.error("Database verification failed")
            
    except Exception as e:
        logger.error(f"Error during database initialization: {str(e)}")
        raise

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created data directory at {data_dir.absolute()}")
    
    # Run the initialization
    asyncio.run(init_database()) 