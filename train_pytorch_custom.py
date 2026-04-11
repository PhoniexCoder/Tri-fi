import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from collections import Counter

class CSIDataset(Dataset):
    """Windowed CSI dataset with optional synthetic "impossible" negatives.

    This mirrors rescue_backend.py preprocessing (64×100 window, mean/std
    normalisation) so the trained model behaves correctly at runtime.
    We also inject synthetic patterns (all‑zeros, flatline, white noise,
    gradients, sine waves) labelled as class 0 (EMPTY) so the network
    learns to reject these instead of hallucinating people.
    """

    def __init__(self, data_dir, window_size=100, add_synthetic_negatives=True, synthetic_per_pattern=500):
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
                    window = frames[i : i + window_size].T  # Becomes (64, 100)

                    # Exact Normalization as used in rescue_backend.py
                    mean = np.mean(window)
                    std = np.std(window) + 1e-6
                    window = (window - mean) / std

                    self.samples.append(window)
                    self.labels.append(label)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")

        # Inject synthetic negative patterns as class 0 (EMPTY)
        if add_synthetic_negatives:
            self._add_synthetic_negatives(window_size=window_size, per_pattern=synthetic_per_pattern)

        # Final stats
        if self.labels:
            counts = Counter(self.labels)
            print("Class distribution (after synthetic negatives):", dict(counts))

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Neural Net expects [Channels, Height, Width] => [1, 64, 100]
        x = torch.tensor(self.samples[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def _add_synthetic_negatives(self, window_size: int, per_pattern: int = 500) -> None:
        """Add synthetic "impossible" CSI matrices labelled as EMPTY (class 0).

        These match the shapes used at inference time: (64, window_size), then
        we apply the same mean/std normalisation as for real data.
        """

        patterns = []

        # 1. Absolute zeros (dead antenna)
        patterns.append(lambda: np.zeros((64, window_size), dtype=np.float32))
        # 2. Flat constant output
        patterns.append(lambda: np.ones((64, window_size), dtype=np.float32) * 55.0)
        # 3. White noise
        patterns.append(lambda: np.random.randn(64, window_size).astype(np.float32))
        # 4. Linear gradient across time
        patterns.append(
            lambda: np.tile(np.linspace(0, 10, window_size, dtype=np.float32), (64, 1))
        )
        # 5. Fast sine wave oscillations
        def _sine():
            t = np.linspace(0, 4 * np.pi, window_size, dtype=np.float32)
            return np.tile(np.sin(t), (64, 1)).astype(np.float32)

        patterns.append(_sine)

        for make_pattern in patterns:
            for _ in range(per_pattern):
                window = make_pattern()
                mean = np.mean(window)
                std = np.std(window) + 1e-6
                window = (window - mean) / std
                self.samples.append(window)
                self.labels.append(0)  # FORCE as EMPTY

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

    # Compute class weights so EMPTY (class 0) is not under-represented.
    # The backend always expects 6 classes (0..5), so force length 6 here
    # even if the current dataset only uses a subset of labels.
    labels_np = np.array(dataset.labels, dtype=np.int64)
    n_classes = 6
    counts = np.bincount(labels_np, minlength=n_classes)
    # Avoid division by zero: for any class with 0 samples, give it the
    # same count as the most frequent class so its weight = ~1.0.
    nonzero = counts > 0
    if not nonzero.any():
        raise RuntimeError("No labels found in dataset; cannot compute class weights.")
    max_count = counts[nonzero].max()
    counts[counts == 0] = max_count
    # Inverse-frequency weighting, normalised
    inv_freq = 1.0 / (counts.astype(np.float32))
    class_weights = (inv_freq / inv_freq.mean()).astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print("Class weights (6 classes):", class_weights)
    
    # Check for hardware acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using Device: {str(device).upper()}")

    model = CSIresnet().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 20
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
    print("\n" + "═"*60)
    print(f"🎉 DONE! Saved {n_classes}-class model to `model.pth`.")
    print(f"The backend will now automatically detect these {n_classes} classes.")
    print("═"*60)

if __name__ == "__main__":
    train()
