from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import logging

from .medical_ethics_assistant import MedicalEthicsAssistant
from .config import SystemConfig

router = APIRouter()
logger = logging.getLogger(__name__)

class TeamsQuery(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    user_id: str
    tenant_id: str

@router.post("/api/teams/query")
async def handle_teams_query(query: TeamsQuery):
    """Handle queries from Teams"""
    try:
        assistant = await MedicalEthicsAssistant.create(SystemConfig())
        
        # Get the ethical guidance
        response = await assistant.get_ethical_guidance(query.query)
        
        # Format for Teams
        return {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {
                    "type": "TextBlock",
                    "text": "Medical Ethics Analysis",
                    "size": "Large",
                    "weight": "Bolder"
                },
                {
                    "type": "TextBlock",
                    "text": response['analysis']['summary'],
                    "wrap": True
                },
                {
                    "type": "FactSet",
                    "facts": [
                        {"title": "Key Point", "value": point}
                        for point in response['analysis']['recommendations']
                    ]
                }
            ],
            "actions": [
                {
                    "type": "Action.Submit",
                    "title": "Ask Follow-up",
                    "data": {
                        "conversation_id": response.get('conversation_id')
                    }
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error processing Teams query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 