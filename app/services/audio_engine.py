import base64
import io
import librosa
import numpy as np
import torch
from app.core.config import settings

class AudioProcessor:
    def process_base64_audio(self, b64_string: str):  # Make sure this name is exact
        try:
            # Clean base64 string
            if "," in b64_string:
                b64_string = b64_string.split(",")[1]
                
            audio_bytes = base64.b64decode(b64_string)
            buffer = io.BytesIO(audio_bytes)
            
            # Load audio
            y, sr = librosa.load(buffer, sr=settings.SAMPLE_RATE)
            
            # Standardize Duration (4 seconds)
            target_samples = settings.SAMPLE_RATE * settings.DURATION
            if len(y) > target_samples:
                y = y[:target_samples]
            else:
                y = np.pad(y, (0, target_samples - len(y)))
                
            # Extract Features
            mel = librosa.feature.melspectrogram(y=y, sr=settings.SAMPLE_RATE, n_mels=128)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-6)
            
            # Shape: [1, 1, 128, 126]
            tensor = torch.FloatTensor(log_mel).unsqueeze(0).unsqueeze(0)
            return tensor
        except Exception as e:
            raise ValueError(f"Invalid Audio Data: {str(e)}")