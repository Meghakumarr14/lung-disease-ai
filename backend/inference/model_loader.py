import os
import torch
import timm
import urllib.request

MODEL_URL = "https://huggingface.co/Meghakumarr14/lung-disease-model/resolve/main/final_best_senet154.pth"
MODEL_PATH = "model/final_best_senet154.pth"

class_names = ['Normal', 'Tuberculosis', 'covid', 'lung-opacity', 'pneumonia']

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from HuggingFace...")
        os.makedirs("model", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded successfully!")

def load_model():
    download_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(
        "senet154",
        pretrained=False,
        num_classes=len(class_names)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device