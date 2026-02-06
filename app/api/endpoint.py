from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.audio_engine import AudioProcessor
from app.services.inference import InferenceService

router = APIRouter()
processor = AudioProcessor()  # This creates the object
inference = InferenceService()

class DetectionRequest(BaseModel):
    audio_b64: str

@router.post("/detect")
async def detect_voice(request: DetectionRequest):
    try:
        # We call the function name we just defined
        tensor = processor.process_base64_audio(request.audio_b64)
        label, score = inference.predict(tensor)
        return {"prediction": label, "confidence": score}
    except Exception as e:
        # This is where your error was appearing
        raise HTTPException(status_code=400, detail=str(e))