import logging
import asyncio
from fastapi import FastAPI
from src.medical_ethics_assistant import MedicalEthicsAssistant
from src.api import setup_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def init_database(app: FastAPI):
    """Initialize database before server starts"""
    try:
        # Create and initialize the assistant
        assistant = await MedicalEthicsAssistant.create()
        
        # Initialize database schema
        await assistant.ethics_db.initialize_database()
        
        # Load initial data
        await assistant.ethics_db.load_initial_data()
        
        # Verify database
        if await assistant.ethics_db.verify_database():
            logger.info("Database initialized and verified successfully")
        else:
            logger.error("Database verification failed")
            raise Exception("Database verification failed")
        
        # Store assistant instance in app state
        app.state.assistant = assistant
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI()
    
    # Setup routes
    setup_routes(app)
    
    # Add startup event to initialize database
    @app.on_event("startup")
    async def startup_event():
        await init_database(app)
    
    return app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8003, reload=True) 