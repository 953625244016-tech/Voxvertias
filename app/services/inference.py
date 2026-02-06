import torch
from app.models.arch import VoiceClassifier
import os

class InferenceService:
    def __init__(self, model_path="weights/best_model.pth"):
        self.device = torch.device("cpu")
        self.model = VoiceClassifier()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, tensor):
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            
        label = "AI_GENERATED" if pred.item() == 1 else "HUMAN"
        return label, float(conf.item())