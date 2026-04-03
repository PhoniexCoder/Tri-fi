import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class CSIDataset(Dataset):
    def __init__(self, data_dir, window_size=100):
        self.samples = []
        self.labels = []
        files = glob.glob(os.path.join(data_dir, "*.npy"))
        
        for f in files:
            try:
                # Expecting filename format: node1_label1_timestamp.npy
                label = int(f.split("_label")[1].split("_")[0])
                frames = np.load(f)  # Shape should equal (Time, 64)
                
                # Split into overlapping windows of size 100
                step = window_size // 2
                for i in range(0, len(frames) - window_size, step):
                    window = frames[i:i+window_size].T  # Becomes (64, 100)
                    
                    # Exact Normalization as used in rescue_backend.py
                    mean = np.mean(window)
                    std = np.std(window) + 1e-6
                    window = (window - mean) / std
                    
                    self.samples.append(window)
                    self.labels.append(label)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Neural Net expects [Channels, Height, Width] => [1, 64, 100]
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Duplicate exact architecture required by rescue_backend.py
class CSIresnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(num_classes=6)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.net(x)

def train():
    print("="*60)
    print("🧠 HACKATHON PYTORCH RESNET18 TRAINING SCRIPT")
    print("="*60)

    dataset = CSIDataset("custom_dataset", window_size=100)
    if len(dataset) == 0:
        print("❌ No training data found in 'custom_dataset/'. Run collect_dataset.py first!")
        return
        
    print(f"✅ Loaded {len(dataset)} windows for training (Batch Size: 32).")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Check for hardware acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using Device: {str(device).upper()}")

    model = CSIresnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 12
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            
        acc = (correct / len(dataset)) * 100
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {acc:.1f}%")

    # Save format exactly matching how rescue_backend.py loads it
    torch.save(model.state_dict(), "model.pth")
    print("\n🎉 DONE! Saved highly-accurate model to `model.pth`.")
    print("You can instantly reboot `rescue_backend.py` to deploy your hackathon model.")

if __name__ == "__main__":
    train()
