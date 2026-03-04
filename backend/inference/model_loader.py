import torch
from model.model_arch import create_model

CLASS_NAMES = ['Normal', 'Tuberculosis', 'covid', 'lung-opacity', 'pneumonia']

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(
        torch.load("model/final_best_senet154.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    return model, device
