# -*- coding: utf-8 -*-
# app.py — CliniScan: Chest X-Ray Classification + Grad-CAM

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import cv2
import pandas as pd
import os
import tempfile

# Optional Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except Exception as e:
    GRADCAM_AVAILABLE = False
    gradcam_import_error = e

# -----------------------
# Utilities: DICOM -> PIL
# -----------------------
import pydicom
from io import BytesIO

def dicom_bytes_to_pil(dcm_bytes):
    """Convert DICOM bytes to a PIL.Image (RGB)."""
    try:
        ds = pydicom.dcmread(BytesIO(dcm_bytes))
        arr = ds.pixel_array
        # Normalize to 0-255
        arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # If single channel, convert to RGB
        if arr_norm.ndim == 2:
            img = Image.fromarray(arr_norm).convert("RGB")
        else:
            img = Image.fromarray(arr_norm).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to parse DICOM: {e}")

# -----------------------
# ResNet Classifier
# -----------------------
from torchvision.models import resnet18
try:
    # newer torchvision uses ResNet18_Weights
    from torchvision.models import ResNet18_Weights
    RESNET_WEIGHTS = ResNet18_Weights.DEFAULT
except Exception:
    RESNET_WEIGHTS = None

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        try:
            if RESNET_WEIGHTS is not None:
                self.model = resnet18(weights=RESNET_WEIGHTS)
            else:
                self.model = resnet18(pretrained=True)
        except Exception:
            self.model = resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# -----------------------
# Model loading helper
# -----------------------
@st.cache_resource
def load_model_from_path(model_path=None, num_classes=2):
    """Load trained model if provided, else fallback ResNet18."""
    model = ResNetClassifier(num_classes=num_classes)
    if model_path:
        try:
            sd = torch.load(model_path, map_location="cpu")
            if isinstance(sd, dict):
                model.load_state_dict(sd)
            else:
                model = sd
            st.success("✅ model.pth loaded.")
            model.eval()
            return model
        except Exception as e:
            st.warning(f"Could not load model.pth: {e}")
            st.info("Falling back to pretrained ResNet18.")
    # fallback
    try:
        if RESNET_WEIGHTS is not None:
            base = resnet18(weights=RESNET_WEIGHTS)
        else:
            base = resnet18(pretrained=True)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        st.info("Using fallback ResNet18 (ImageNet weights).")
        base.eval()
        return base
    except Exception as e:
        raise RuntimeError(f"Failed to create model: {e}")

# -----------------------
# Preprocessing
# -----------------------
from torchvision import transforms as T
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="CliniScan — Chest X-Ray (Grad-CAM)", layout="wide")
st.title("🩻 CliniScan — Chest X-Ray Classification + Grad-CAM")

with st.sidebar:
    st.header("Options")
    num_classes = st.number_input("Number of classes (final layer)", value=2, min_value=2, max_value=20)
    upload_model = st.file_uploader("Upload model.pth (optional)", type=["pt", "pth"])
    show_gradcam = st.checkbox("Enable Grad-CAM visualization", value=True)
    show_probs = st.checkbox("Show class probabilities", value=True)

# Handle uploaded model
model_path = None
if upload_model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tfile.write(upload_model.getvalue())
    tfile.flush()
    model_path = tfile.name

model = load_model_from_path(model_path=model_path, num_classes=int(num_classes))

# Image uploader
uploaded = st.file_uploader("Upload an X-ray image or DICOM (.dcm)", type=["png","jpg","jpeg","dcm"])
if uploaded is None:
    st.info("Upload a chest X-ray image (jpg/png) or a DICOM (.dcm) file to run prediction.")
else:
    try:
        filename = uploaded.name.lower()
        if filename.endswith(".dcm"):
            img = dicom_bytes_to_pil(uploaded.getvalue())
        else:
            img = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

        st.image(img, caption="Uploaded image", use_column_width=True)

        # Preprocess and predict
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()

        class_names = [f"Class_{i}" for i in range(int(num_classes))]
        if int(num_classes) == 2:
            class_names = ["Normal","Pneumonia"]

        if show_probs:
            st.subheader("Prediction")
            st.write(f"**{class_names[pred_class]}** — {probs[pred_class].item()*100:.2f}% confidence")
            chart_data = {class_names[i]: float(probs[i].item()) for i in range(len(probs))}
            st.bar_chart(pd.DataFrame.from_dict(chart_data, orient="index", columns=["probability"]))

        # -----------------------
        # Grad-CAM Visualization
        # -----------------------
        if show_gradcam:
            if not GRADCAM_AVAILABLE:
                st.error("pytorch-grad-cam is not installed.")
                st.exception(gradcam_import_error)
            else:
                try:
                    if hasattr(model, "model"):  # ResNetClassifier wrapper
                        target_layers = [model.model.layer4[-1]]
                    else:
                        target_layers = [model.layer4[-1]]

                    # ✅ No use_cuda argument (new API)
                    cam = GradCAM(model=model, target_layers=target_layers)

                    rgb_img = np.array(img.resize((224,224))) / 255.0
                    grayscale_cam = cam(input_tensor=input_tensor,
                                        targets=[ClassifierOutputTarget(pred_class)])[0]

                    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                    st.subheader("Grad-CAM")
                    st.image([rgb_img, visualization],
                             caption=["Original", "Grad-CAM visualization"],
                             use_column_width=True)
                except Exception as e:
                    st.error(f"Grad-CAM failed: {e}")

    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

       
