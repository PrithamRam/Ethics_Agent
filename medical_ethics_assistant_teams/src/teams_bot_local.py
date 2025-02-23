from botbuilder.core import TurnContext, ActivityHandler
from botbuilder.schema import Activity, ActivityTypes
import json
import asyncio
from .medical_ethics_assistant import MedicalEthicsAssistant
from .config import SystemConfig

class LocalTeamsBot(ActivityHandler):
    def __init__(self):
        self.assistant = None
        
    async def initialize(self):
        """Initialize the medical ethics assistant"""
        self.assistant = await MedicalEthicsAssistant.create(SystemConfig())
    
    async def on_message_activity(self, turn_context: TurnContext):
        """Handle incoming messages"""
        if not self.assistant:
            await self.initialize()
            
        query = turn_context.activity.text
        try:
            # Get ethical guidance
            response = await self.assistant.get_ethical_guidance(query)
            
            # Create adaptive card response
            card = {
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
                    }
                ]
            }
            
            # Send response
            await turn_context.send_activity(Activity(
                type=ActivityTypes.message,
                attachment_layout="list",
                attachments=[{
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": card
                }]
            ))
            
        except Exception as e:
            await turn_context.send_activity(f"Error: {str(e)}") 