from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from .teams_bot_local import LocalTeamsBot
from pyngrok import ngrok
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
bot = LocalTeamsBot()

# Configure ngrok
ngrok.set_auth_token(os.getenv('NGROK_AUTH_TOKEN'))

@app.on_event("startup")
async def startup_event():
    # Start ngrok tunnel
    try:
        url = ngrok.connect(8000)
        print(f"\nPublic URL: {url}")
        print(f"Teams Message Endpoint: {url}/api/messages\n")
        print("Use this URL in your Teams manifest")
    except Exception as e:
        print(f"Error starting ngrok: {str(e)}")

@app.post("/api/messages")
async def messages(request: Request):
    """Handle incoming messages from Teams"""
    body = await request.json()
    
    # Handle message
    response = await bot.on_message_activity(body)
    return JSONResponse(content=response)

@app.get("/")
async def home():
    """Home page with instructions"""
    return {
        "message": "Medical Ethics Assistant Bot",
        "instructions": [
            "1. Use ngrok to create a public URL",
            "2. Update the messaging endpoint in Teams",
            "3. Install the app in Teams"
        ]
    } 