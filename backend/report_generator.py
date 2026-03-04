from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image


def generate_medical_report(data):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40

    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "Chest X-ray Analysis Report")

    y -= 30
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Patient Details")

    pdf.setFont("Helvetica", 11)
    y -= 20
    pdf.drawString(60, y, f"Name: {data['name']}")
    y -= 18
    pdf.drawString(60, y, f"Age: {data['age']}")
    y -= 18
    pdf.drawString(60, y, f"Gender: {data['gender']}")

    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Diagnosis Result")

    pdf.setFont("Helvetica", 11)
    y -= 20
    pdf.drawString(60, y, f"Predicted Disease: {data['predicted_class']}")

    y -= 25
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(60, y, "Confidence Scores:")

    pdf.setFont("Helvetica", 11)
    for disease, prob in data["probabilities"].items():
        y -= 18
        pdf.drawString(70, y, f"{disease}: {prob * 100:.2f}%")

    # Grad-CAM Image
    if data.get("gradcam_image"):
        y -= 30
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Grad-CAM Visualization")

        img_data = base64.b64decode(data["gradcam_image"])
        image = Image.open(BytesIO(img_data))
        image_path = "temp_gradcam.png"
        image.save(image_path)

        pdf.drawImage(image_path, 60, y - 220, width=200, height=200)

    y -= 260
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(
        50,
        y,
        "Disclaimer: This AI-generated report is for educational purposes only "
        "and must not be considered a medical diagnosis."
    )

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return buffer
