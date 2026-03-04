import gradio as gr
import torch
import timm
from torchvision import transforms
from PIL import Image

# --- Class names (must match training order) ---
class_names = ['Normal', 'Tuberculosis', 'covid', 'lung-opacity', 'pneumonia']

# --- Load trained SENet154 model ---
model = timm.create_model('senet154', pretrained=False, num_classes=len(class_names))
model.load_state_dict(torch.load("model/final_best_senet154.pth", map_location='cpu'))
model.eval()

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# --- Prediction function ---
def predict(image):
    img = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        return {cls: float(prob) for cls, prob in zip(class_names, probs)}

# --- UI ---
gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Lung Disease Detection (SENet154)",
    description="Upload a chest X-ray image to classify it as Normal, Tuberculosis, Covid, Lung Opacity, or Pneumonia."
).launch(share=True)
