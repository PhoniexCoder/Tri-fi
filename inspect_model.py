import torch

try:
    model = torch.load(r'examples\model.pth', map_location='cpu', weights_only=False)
    print(f"Type of loaded model: {type(model)}")
    if isinstance(model, dict):
        print("Keys in state_dict:")
        for k, v in model.items():
            print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
    else:
        print("Model:")
        print(model)
except Exception as e:
    print(f"Failed to load model natively: {e}")
