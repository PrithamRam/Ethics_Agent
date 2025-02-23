import uvicorn
import logging
from src.medical_ethics_assistant import MedicalEthicsAssistant
from src.config import SystemConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Medical Ethics Assistant server...")
    logger.info("Server will be available at http://127.0.0.1:8000")
    logger.info("Press Ctrl+C to stop the server")
    
    # Initialize the assistant
    assistant = MedicalEthicsAssistant.create(SystemConfig())
    
    # Run the server
    uvicorn.run(
        "src.api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    ) 