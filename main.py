import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
import uvicorn
from app.api.endpoint import router as api_router

app = FastAPI()

@app.get("/")
def health():
    return {"status": "online", "engine": "LightCNN-V2"}

app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)