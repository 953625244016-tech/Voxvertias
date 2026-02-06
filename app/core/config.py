import os

class Settings:
    PROJECT_NAME: str = "AI Voice Detector"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Audio Settings
    SAMPLE_RATE: int = 16000
    DURATION: int = 4
    
    # Path logic
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODEL_PATH: str = os.path.join(BASE_DIR, "weights", "best_model.pth")

settings = Settings()