from inference.predict import predict_image

result = predict_image("sample_xray.png")

print("Prediction probabilities:")
for disease, prob in result.items():
    print(f"{disease}: {prob:.4f}")
