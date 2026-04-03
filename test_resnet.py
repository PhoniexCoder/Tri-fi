import torch
import torch.nn as nn
import torchvision.models as models

class CSIresnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(num_classes=6)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    def forward(self, x):
        return self.net(x)

model = CSIresnet()
try:
    state_dict = torch.load('model.pth', map_location='cpu', weights_only=False)
    
    # Check if we need to remove 'net.' from keys
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('net.'):
            # Because our class has self.net, the keys match perfectly!
            new_state_dict[k] = v
        else:
            new_state_dict['net.'+k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    
    # Empty room typically has low noise floor (-100 dBm) mapped to maybe -1 or 0 in typical scaling.
    # We will try passing zeros.
    x_empty = torch.zeros(1, 1, 64, 100)
    with torch.no_grad():
        out = model(x_empty)
    print("Output across 6 classes for zeros:", out)
    print("Argmax:", out.argmax().item())

except Exception as e:
    print(f"Failed: {e}")
