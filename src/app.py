from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize assistant and store in app state
    from src.medical_ethics_assistant import MedicalEthicsAssistant
    logger.info("Initializing Medical Ethics Assistant...")
    app.state.assistant = await MedicalEthicsAssistant.create()
    logger.info("Medical Ethics Assistant initialized")
    yield
    # Cleanup (if needed)
    logger.info("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(title="Medical Ethics Assistant", lifespan=lifespan) 