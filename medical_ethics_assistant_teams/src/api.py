from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.medical_ethics_assistant import MedicalEthicsAssistant
import logging
import uuid

app = FastAPI()
logger = logging.getLogger(__name__)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Store active conversations
active_conversations = {}

class Query(BaseModel):
    query: str
    conversation_id: str = None

class FollowUp(BaseModel):
    question: str
    conversation_id: str = None

async def get_assistant(conversation_id: str = None):
    """Get or create an assistant instance"""
    if conversation_id and conversation_id in active_conversations:
        return active_conversations[conversation_id]
    
    assistant = await MedicalEthicsAssistant.create()
    if conversation_id:
        active_conversations[conversation_id] = assistant
    return assistant

@app.get("/")
async def get_home():
    """Serve the main page"""
    return templates.TemplateResponse("query_page.html", {"request": {}})

@app.post("/api/ethical-guidance")
async def get_ethical_guidance(query: Query):
    """Handle the main ethical guidance query"""
    try:
        conversation_id = query.conversation_id or str(uuid.uuid4())
        assistant = await get_assistant(conversation_id)
        
        # Get the ethical guidance
        response = await assistant.get_ethical_guidance(query.query)
        
        # Render the response HTML
        html = templates.get_template("ethics_response.html").render({
            "query": query.query,
            "analysis": response["analysis"],
            "references": response["references"],
            "conversation_id": conversation_id
        })
        
        return JSONResponse({
            "status": "success",
            "html": html,
            "conversation_id": conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        error_html = templates.get_template("error.html").render({
            "error": str(e),
            "request": {}
        })
        return JSONResponse({
            "status": "error",
            "html": error_html,
            "error": str(e)
        })

@app.post("/api/follow-up")
async def handle_follow_up(follow_up: FollowUp):
    """Handle follow-up questions"""
    try:
        if not follow_up.conversation_id:
            raise HTTPException(status_code=400, detail="No conversation ID provided")
            
        assistant = await get_assistant(follow_up.conversation_id)
        if not assistant:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        # Get the follow-up response
        response = await assistant.get_follow_up_response(follow_up.question)
        
        # Render the follow-up response HTML
        html = templates.get_template("follow_up_response.html").render({
            "question": follow_up.question,
            "analysis": response["analysis"],
            "references": response.get("references", []),
            "conversation_id": follow_up.conversation_id
        })
        
        return JSONResponse({
            "status": "success",
            "html": html,
            "conversation_id": follow_up.conversation_id
        })
        
    except Exception as e:
        logger.error(f"Error processing follow-up: {str(e)}")
        return JSONResponse({
            "status": "error",
            "html": templates.get_template("error.html").render({
                "error": str(e),
                "request": {}
            }),
            "error": str(e)
        })

@app.get("/api/sample-question")
async def get_sample_question():
    """Get a sample question based on database content"""
    try:
        assistant = await MedicalEthicsAssistant.create()
        question = await assistant.ethics_db.get_sample_question()
        return {"question": question}
    except Exception as e:
        logger.error(f"Error getting sample question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/initialize")
async def initialize_database():
    """Initialize the database with ventilator ethics paper"""
    try:
        assistant = await MedicalEthicsAssistant.create()
        success = await assistant.initialize_database()
        if success:
            return {"status": "success", "message": "Database initialized with ventilator ethics paper"}
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize database")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database-status")
async def get_database_status():
    """Get current database status"""
    try:
        assistant = await MedicalEthicsAssistant.create()
        papers = await assistant.ethics_db.list_all_papers()
        return {
            "status": "success",
            "paper_count": len(papers),
            "papers": papers
        }
    except Exception as e:
        logger.error(f"Error getting database status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 