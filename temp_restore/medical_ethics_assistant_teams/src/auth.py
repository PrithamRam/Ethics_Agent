from fastapi import Depends, HTTPException
from fastapi.security import OAuth2AuthorizationCodeBearer
from msal import ConfidentialClientApplication
import jwt

class TeamsAuth:
    def __init__(self):
        self.client_id = "your_client_id"
        self.client_secret = "your_client_secret"
        self.tenant_id = "your_tenant_id"
        
    async def validate_token(self, token: str):
        try:
            # Decode and validate Teams token
            decoded = jwt.decode(
                token,
                verify=False  # In production, verify with public key
            )
            return decoded
        except Exception as e:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def get_current_user(self, token: str = Depends(OAuth2AuthorizationCodeBearer)):
        user_data = await self.validate_token(token)
        return user_data 