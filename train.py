import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import glob
import base64
from app.models.arch import VoiceClassifier
from app.services.audio_engine import AudioProcessor

class VoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.processor = AudioProcessor()
        self.samples = []
        # Support both mp3 and wav
        for label, folder in [(0, "human"), (1, "ai")]:
            target_path = os.path.join(root_dir, folder)
            files = glob.glob(os.path.join(target_path, "*.mp3")) + \
                    glob.glob(os.path.join(target_path, "*.wav"))
            for f in files:
                self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        tensor = self.processor.process_base64_audio(b64)
        return tensor.squeeze(0), label

def train():
    os.makedirs("weights", exist_ok=True)
    device = torch.device("cpu")
    model = VoiceClassifier().to(device)
    
    # This line was triggering the error because of 'transformers'
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    train_data = VoiceDataset("data/train")
    if len(train_data) == 0:
        print("❌ Error: No files found in data/train/human or data/train/ai")
        return

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    
    print(f"🚀 Training on {len(train_data)} files...")
    for epoch in range(10):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/10 | Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "weights/best_model.pth")
    print("✅ Training Complete!")

if __name__ == "__main__":
    train()