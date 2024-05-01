from fastapi import HTTPException, Header, Security
from fastapi.security import APIKeyHeader

from dotenv import load_dotenv
import os


load_dotenv()
API_KEY = os.getenv("API_KEY")
 
api_key_header = APIKeyHeader(name="api_key", auto_error=False)


def get_api_key(api_key_header: str = Security(api_key_header)):   
    print(api_key_header)
    if api_key_header == API_KEY:
        return True
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")
