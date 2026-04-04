import os
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

# Mirror the exact architecture from the backend
class CSICNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(num_classes=6)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.net(x)

def run_test(model, name, mock_matrix):
    # Identical normalization logic used in rescue_backend.py
    mean = np.mean(mock_matrix)
    std = np.std(mock_matrix) + 1e-6
    norm = (mock_matrix - mean) / std

    tensor_in = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor_in)
        probs = torch.softmax(out, dim=1).numpy()[0]

    print(f"--- TEST: {name} ---")
    # Print probabilities for all 6 classes for full visibility.
    # In the current training setup, class 0 = EMPTY, and any
    # non-zero class (1-5) means HUMAN PRESENT (binary).
    for idx, p in enumerate(probs):
        if idx == 0:
            label = "Empty / No Human"
        else:
            label = f"Human Present (class {idx})"
        print(f"  Class {idx}: {p*100:5.1f}%  -> {label}")

    argmax = int(np.argmax(probs))
    if argmax == 0:
        print("Result: ✅ Correctly rejected as EMPTY/NOISE.")
    else:
        print(f"Result: 🚨 BIAS ALARM! Falsely predicted human presence (Class {argmax}).")
    print()

def main():
    model_path = "model.pth"
    if not os.path.exists(model_path):
        print("model.pth not found! Run train_pytorch_custom.py first.")
        return
        
    model = CSICNN()
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("="*60)
    print("🧠 CSI MODEL: ANTI-BIAS / HALLUCINATION TEST")
    print("Feeding mathematically weird signals into the Neural Network.")
    print("If it guesses a Human is present, your model is biased!")
    print("="*60)

    # 1. Absolute Zeros (Dead Antenna)
    run_test(model, "Absolute Space Silence (All Zeros)", np.zeros((64, 100)))
    
    # 2. Perfect Constant (Crashed Antenna Outputting 55 flat)
    run_test(model, "Flatline Hardware Crash (All 55s)", np.ones((64, 100)) * 55.0)

    # 3. Aggressive White Noise (Heavy radio interference)
    run_test(model, "Aggressive RF Static / White Noise", np.random.randn(64, 100))

    # 4. Linear Gradient (Smooth baseline shifts)
    grad = np.tile(np.linspace(0, 10, 100), (64, 1))
    run_test(model, "Temperature Drift (Linear Gradient)", grad)
    
    # 5. Fast Sine Wave Oscillations (Microwave oven running nearby)
    t = np.linspace(0, 4*np.pi, 100)
    sine = np.tile(np.sin(t), (64, 1))
    run_test(model, "50Hz Magnetron Interference (Sine Wave)", sine)

if __name__ == "__main__":
    main()
