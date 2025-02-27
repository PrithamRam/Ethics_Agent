import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import logging
from dotenv import load_dotenv
from src.api import router
from src.app import app  # Import the shared app instance
import asyncio
from src.ethics_db import EthicsDB
from pathlib import Path
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from src.medical_ethics_assistant import MedicalEthicsAssistant
import signal
import sys
import os

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

# Add before app initialization
warnings.filterwarnings('ignore', category=NotOpenSSLWarning)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include the API router
app.include_router(router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    html_file = Path("static/index.html")
    return html_file.read_text()

async def cleanup(signal_name=None):
    """Cleanup resources"""
    logger.info("Cleaning up resources...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Cleanup complete")

def handle_exit(signum, frame):
    logger.info(f"Received signal {signum}")
    asyncio.create_task(cleanup())
    sys.exit(0)

@app.on_event("startup")
async def startup_event():
    """Initialize the assistant and database on startup"""
    try:
        logger.info("\n" + "="*50)
        logger.info("Starting initialization...")
        
        # Initialize database
        logger.info("Initializing database...")
        db = EthicsDB()
        
        # Initialize Elasticsearch and load papers
        success = await db.initialize_database()
        if not success:
            raise Exception("Failed to initialize database")
        
        # Initialize Medical Ethics Assistant
        logger.info("Initializing Medical Ethics Assistant...")
        app.state.assistant = await MedicalEthicsAssistant.create()
        app.state.db = db
        
        logger.info("Initialization complete")
        logger.info("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Kill any existing process on port 8005
    os.system("kill -9 $(lsof -t -i:8005) 2>/dev/null || true")

    # Run startup
    asyncio.run(startup_event())

    # Start server
    uvicorn.run(
        "src.app:app",
        host="127.0.0.1",
        port=8005,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main() 