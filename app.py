import streamlit as st
import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
from streamlit_drawable_canvas import st_canvas

class MNISTModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.block1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.block2 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.50),
        nn.Linear(in_features=128*7*7, out_features=10)
    )

  def forward(self, x : torch.Tensor):
    x = self.block1(x)
    x = self.block2(x)
    x = self.classifier(x)
    return x


model = MNISTModel()
model.load_state_dict(torch.load("D:/Project1/models/digit_recognition_0.pth"))
model.eval()

st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✍️", layout="centered")

st.markdown("<h1 style='text-align:center;'>✍️ Handwritten Digit Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Draw a digit (0–9) below and click <b>Predict</b> to see the model's guess.</p>", unsafe_allow_html=True)

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas"

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key
)

col1, col2 = st.columns(2)

with col1:
    predict_clicked = st.button("🔍 Predict", use_container_width=True)

with col2:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    st.session_state.canvas_key = "canvas_" + str(np.random.randint(0, 10000))
    st.rerun()


if predict_clicked:
    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert('L')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)
            predicted_digit = torch.argmax(pred, dim=1).item()

        st.markdown("<h3 style='text-align:center;'>🎯 Prediction Result</h3>", unsafe_allow_html=True)
        col3, col4 = st.columns([1,1])
        with col3:
            st.image(img, caption="Processed (28×28)", width=100)
        with col4:
            st.success(f"Predicted Digit: **{predicted_digit}**")