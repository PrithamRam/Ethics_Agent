import uvicorn
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

# Suppress verbose logs
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    logger.info("Starting Medical Ethics Assistant...")
    
    # Initialize database
    from src.ethics_database import EthicsDatabase
    db = EthicsDatabase.initialize_database_if_needed()
    
    # Run the server
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8003,
        reload=True
    ) 