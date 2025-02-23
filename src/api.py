from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel, validator
from src.medical_ethics_assistant import MedicalEthicsAssistant
import logging
import uuid
from typing import Optional, Dict, List, Any
import os
import io
import csv
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
from .ethics_database import EthicsDatabase
from .pubmed_handler import PubMedHandler
from .response_parser import ResponseParser

app = FastAPI(
    title="Medical Ethics AI Assistant",
    description="API for querying medical ethics papers and getting AI-generated responses",
    version="1.0.0"
)
logger = logging.getLogger(__name__)

# Store active conversations
active_conversations = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory with cache busting
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Initialize components
db = EthicsDatabase.initialize_database_if_needed()
pubmed_handler = PubMedHandler()
response_parser = ResponseParser()

class Query(BaseModel):
    text: str
    max_results: int = 10

    @validator('text')
    def question_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty')
        if len(v) < 10:
            raise ValueError('Question must be at least 10 characters long')
        return v

class FollowUp(BaseModel):
    question: str
    conversation_id: str

    @validator('question')
    def question_must_not_be_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Question cannot be empty')
        if len(v) < 10:
            raise ValueError('Question must be at least 10 characters long')
        return v

    @validator('conversation_id')
    def conversation_id_must_be_valid(cls, v):
        if not v:
            raise ValueError('Conversation ID is required')
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError('Invalid conversation ID format')
        return v

async def get_assistant(conversation_id: str = None):
    """Get or create an assistant instance"""
    if conversation_id and conversation_id in active_conversations:
        return active_conversations[conversation_id]
    
    assistant = await MedicalEthicsAssistant.create()
    if conversation_id:
        active_conversations[conversation_id] = assistant
    return assistant

@app.post("/api/ethical-guidance")
async def get_ethical_guidance(request: Request):
    try:
        data = await request.json()
        logger.info(f"Raw request data: {data}")
        
        # Parse query and max_results
        query = data.get('text', '')
        max_results = int(data.get('max_results', 10))
        logger.info(f"Parsed query: text='{query}' max_results={max_results}")
        
        # Search for relevant papers
        papers = db.search_papers(query, max_results)
        logger.info(f"Found {len(papers)} papers")
        
        # Generate response
        response = response_parser.generate_response(query, papers)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}

@app.post("/api/follow-up")
async def get_follow_up(follow_up: FollowUp):
    """Handle follow-up questions"""
    try:
        # Validate conversation exists
        if follow_up.conversation_id not in active_conversations:
            return JSONResponse({
                "status": "error",
                "error": "Conversation not found"
            }, status_code=status.HTTP_404_NOT_FOUND)
        
        # Get assistant and response
        try:
            assistant = await get_assistant(follow_up.conversation_id)
            response = await assistant.get_follow_up_response(follow_up.question)
        except Exception as e:
            logger.error(f"Error getting follow-up response: {str(e)}", exc_info=True)
            return JSONResponse({
                "status": "error",
                "error": "Failed to generate follow-up response"
            }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Return validated response
        return JSONResponse({
            "status": "success",
            "literature_analysis": response.get('literature_analysis', {
                "summary": "No literature analysis available",
                "recommendations": [],
                "concerns": [],
                "mitigation_strategies": []
            }),
            "ai_analysis": response.get('ai_analysis', {
                "summary": "No AI analysis available",
                "recommendations": [],
                "concerns": [],
                "mitigation_strategies": []
            }),
            "references": response.get('references', []),
            "conversation_id": follow_up.conversation_id
        })
        
    except ValueError as e:
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        logger.error(f"Unexpected error in follow-up: {str(e)}", exc_info=True)
        return JSONResponse({
            "status": "error",
            "error": "An unexpected error occurred"
        }, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

@app.get("/api/ethical-guidance/csv")
async def get_ethical_guidance_csv(question: str):
    """Return ethical guidance as CSV"""
    try:
        logger.info(f"Starting CSV generation for question: {question}")
        
        conversation_id = str(uuid.uuid4())
        assistant = await get_assistant(conversation_id)
        response = await assistant.get_ethical_guidance(question)
        
        logger.info(f"Got response: {response}")
        
        # Create CSV file in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write headers
        writer.writerow(['Source', 'Category', 'Content'])
        
        # Write Literature Analysis
        writer.writerow(['Literature Analysis', 'Summary', response.get('literature_analysis', {}).get('summary', 'No literature analysis available')])
        
        for rec in response.get('literature_analysis', {}).get('recommendations', []):
            writer.writerow(['Literature Analysis', 'Recommendation', rec])
            
        for concern in response.get('literature_analysis', {}).get('concerns', []):
            writer.writerow(['Literature Analysis', 'Concern', concern])
            
        for strategy in response.get('literature_analysis', {}).get('mitigation_strategies', []):
            writer.writerow(['Literature Analysis', 'Mitigation Strategy', strategy])
            
        # Write AI Analysis
        writer.writerow(['AI Analysis', 'Summary', response.get('ai_analysis', {}).get('summary', 'No AI analysis available')])
        
        for rec in response.get('ai_analysis', {}).get('recommendations', []):
            writer.writerow(['AI Analysis', 'Recommendation', rec])
            
        for concern in response.get('ai_analysis', {}).get('concerns', []):
            writer.writerow(['AI Analysis', 'Concern', concern])
            
        for strategy in response.get('ai_analysis', {}).get('mitigation_strategies', []):
            writer.writerow(['AI Analysis', 'Mitigation Strategy', strategy])
            
        # Write References
        for ref in response.get('references', []):
            writer.writerow(['Reference', 'Title', ref.get('title', '')])
            writer.writerow(['Reference', 'Abstract', ref.get('abstract', '')])
            writer.writerow(['Reference', 'Citation', ref.get('citation', '')])
            writer.writerow(['Reference', 'Ethical Considerations', ', '.join(ref.get('ethical_considerations', []))])
            writer.writerow(['Reference', 'Keywords', ', '.join(ref.get('keywords', []))])
            writer.writerow(['Reference', 'Relevance Score', str(ref.get('relevance_score', ''))])
        
        # Get the final CSV content
        csv_content = output.getvalue()
        logger.info(f"Generated CSV content: {csv_content}")
        
        # Return CSV file with explicit encoding
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={
                'Content-Disposition': f'attachment; filename="ethical_guidance_{conversation_id}.csv"',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}", exc_info=True)
        error_csv = "Source,Category,Content\nError,Error Message," + str(e)
        return StreamingResponse(
            iter([error_csv]),
            media_type="text/csv",
            headers={
                'Content-Disposition': 'attachment; filename="error.csv"',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )

@app.get("/")
async def get_home():
    """Serve the input form HTML with cache busting"""
    return FileResponse('static/index.html', headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    })

@app.post("/search")
async def search_papers(query: Query):
    papers = db.search_papers(query.text, limit=query.max_results)
    return {
        "papers": papers,
        "total_found": len(papers)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/version")
async def get_version():
    """Check the version of index.html being served"""
    with open('static/index.html', 'r') as f:
        content = f.read()
        # Check if the file contains the correct request format
        if 'body: JSON.stringify({ text: query' in content:
            return {"status": "correct", "version": "new"}
        return {"status": "incorrect", "version": "old"} 