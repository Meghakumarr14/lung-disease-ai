import base64
import cv2
import numpy as np

# Corrected imports
from backend.explainability.gradcam import generate_gradcam
from backend.inference.predict import predict_with_top_class
from backend.inference.predict import predict_image
from backend.report_generator import generate_medical_report

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from PIL import Image
import io

app = FastAPI(title="Lung Disease Classification API")

# Allow frontend access later (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "API is running"}


# -----------------------------
# Prediction API
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    temp_path = "temp.jpg"
    image.save(temp_path)

    result = predict_image(temp_path)

    return {"predictions": result}


# -----------------------------
# GradCAM API
# -----------------------------
@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    temp_path = "temp_gradcam.jpg"
    image.save(temp_path)

    heatmap, original_image = generate_gradcam(temp_path)

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(original_image.resize((224, 224)))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return {"gradcam_image": encoded_image}


# -----------------------------
# Combined Analysis API
# -----------------------------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    temp_path = "temp_analyze.jpg"
    image.save(temp_path)

    # Prediction
    prediction_result = predict_with_top_class(temp_path)

    # GradCAM
    heatmap, original_image = generate_gradcam(
        temp_path,
        prediction_result["class_index"]
    )

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = np.array(original_image.resize((224, 224)))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    _, buffer = cv2.imencode(".jpg", overlay)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return {
        "predicted_class": prediction_result["predicted_class"],
        "probabilities": prediction_result["probabilities"],
        "gradcam_image": encoded_image
    }


# -----------------------------
# PDF Report Generator
# -----------------------------
@app.post("/generate-report")
async def generate_report(payload: dict):

    pdf_buffer = generate_medical_report(payload)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=medical_report.pdf"
        },
    )


# -----------------------------
# Medical Chatbot
# -----------------------------
@app.post("/chat")
async def medical_chat(payload: dict):

    disease = payload.get("disease")
    question = payload.get("question")

    medical_knowledge = {

        "Pneumonia": {
            "description": "Pneumonia is an infection that inflames air sacs in one or both lungs.",
            "symptoms": "Common symptoms include cough, fever, chills, and difficulty breathing.",
            "precautions": "Rest, hydration, prescribed antibiotics (if bacterial), and medical supervision.",
            "severity": "Severity depends on age and immune status."
        },

        "Tuberculosis": {
            "description": "Tuberculosis is a bacterial infection that primarily affects the lungs.",
            "symptoms": "Persistent cough, weight loss, night sweats, fever.",
            "precautions": "Long-term antibiotic treatment and medical monitoring.",
            "severity": "Untreated TB can be serious but is treatable."
        },

        "covid": {
            "description": "COVID-19 is a viral respiratory infection caused by SARS-CoV-2.",
            "symptoms": "Fever, cough, fatigue, breathing difficulty.",
            "precautions": "Isolation, hydration, oxygen monitoring.",
            "severity": "Severity varies by patient condition."
        },

        "lung-opacity": {
            "description": "Lung opacity refers to dense areas seen in lung imaging.",
            "symptoms": "Depends on underlying condition.",
            "precautions": "Further medical evaluation required.",
            "severity": "Requires clinical diagnosis."
        },

        "Normal": {
            "description": "The chest X-ray appears normal with no abnormalities.",
            "symptoms": "No pathological signs detected.",
            "precautions": "Maintain healthy lifestyle.",
            "severity": "No medical concern detected."
        }
    }

    response_text = "Please consult a healthcare professional."

    if disease in medical_knowledge:

        info = medical_knowledge[disease]

        if "symptom" in question.lower():
            response_text = info["symptoms"]

        elif "precaution" in question.lower():
            response_text = info["precautions"]

        elif "serious" in question.lower() or "severity" in question.lower():
            response_text = info["severity"]

        else:
            response_text = info["description"]

    return {
        "response": response_text + "\n\n⚠ This AI system provides educational information only."
    }