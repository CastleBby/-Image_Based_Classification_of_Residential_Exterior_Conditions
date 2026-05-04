import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import get_model

# Load model
model = get_model(num_classes=3)
model.load_state_dict(torch.load("outputs/models/combined_model.pth", map_location="cpu"))
model.eval()

# Class names (must match your dataset order)
classes = ["average", "poor", "well_maintained"]

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("Residential Exterior Condition Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)

    prediction = classes[pred.item()]

    st.subheader(f"Prediction: {prediction}")