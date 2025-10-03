# -*- coding: utf-8 -*-
# app.py — CliniScan: Chest X-Ray Classification + Grad-CAM

# app.py
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
import plotly.express as px

# Optional imports that may fail if not installed
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except Exception as e:
    GRADCAM_AVAILABLE = False
    gradcam_import_error = e

# -----------------------
# DICOM -> PIL utility
# -----------------------
import pydicom
from io import BytesIO

def dicom_bytes_to_pil(dcm_bytes):
    """Convert DICOM bytes to a PIL.Image (RGB)."""
    ds = pydicom.dcmread(BytesIO(dcm_bytes))
    arr = ds.pixel_array
    arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    if arr_norm.ndim == 2:
        img = Image.fromarray(arr_norm).convert("RGB")
    else:
        img = Image.fromarray(arr_norm).convert("RGB")
    return img

# -----------------------
# Dataset class (kept for compatibility)
# -----------------------
from torch.utils.data import Dataset
class VinDrCXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, task='classification'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.task = task

        self.labels = sorted(df['label'].unique())
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}

        self.grouped_images = self.df.groupby('image_id')
        self.image_ids = list(self.grouped_images.groups.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert("RGB")
        annotations = self.grouped_images.get_group(img_id)

        if self.task == 'classification':
            label_vector = torch.zeros(len(self.labels), dtype=torch.float)
            for _, row in annotations.iterrows():
                label_vector[self.label_to_id[row['label']]] = 1.0

            if self.transform:
                image = self.transform(image)

            return image, label_vector

        elif self.task == 'detection':
            boxes, labels = [], []
            for _, row in annotations.iterrows():
                boxes.append([row['x_min'], row['y_min'], row['x_max'], row['y_max']])
                labels.append(self.label_to_id[row['label']])

            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64)
            }

            if self.transform:
                image = self.transform(image)

            return image, target

# -----------------------
# ResNetClassifier
# -----------------------
from torchvision.models import resnet18
try:
    from torchvision.models import ResNet18_Weights
    RESNET_WEIGHTS = ResNet18_Weights.DEFAULT
except Exception:
    RESNET_WEIGHTS = None

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        if RESNET_WEIGHTS is not None:
            self.model = resnet18(weights=RESNET_WEIGHTS)
        else:
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
# Custom CSS
# -----------------------
st.markdown(
    """
    <style>
    .big-title {
        font-size:36px !important;
        font-weight:bold;
        color:#00BFFF;
        text-align:center;
    }
    .footer {
        font-size:14px;
        text-align:center;
        color:gray;
        margin-top:50px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="CliniScan — Chest X-Ray (Grad-CAM)", layout="wide")
st.markdown('<p class="big-title">🩻 CliniScan — Chest X-Ray Classification + Grad-CAM</p>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-lighttext.png", width=120)
    st.markdown("### Upload Settings")
    num_classes = st.number_input("Number of classes", value=2, min_value=2, max_value=20)
    upload_model = st.file_uploader("Upload model (.pth)", type=["pt", "pth"])
    show_gradcam = st.checkbox("Enable Grad-CAM", value=True)
    show_probs = st.checkbox("Show probabilities", value=True)

model_path = None
if upload_model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
    tfile.write(upload_model.getvalue())
    tfile.flush()
    model_path = tfile.name

model = load_model_from_path(model_path=model_path, num_classes=int(num_classes))

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

        # Preprocess + predict
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

            # Prettier Plotly chart
            chart_data = pd.DataFrame({
                "Class": class_names,
                "Probability": [float(probs[i].item()) for i in range(len(probs))]
            })
            fig = px.bar(
                chart_data,
                x="Class",
                y="Probability",
                color="Class",
                text="Probability",
                color_discrete_sequence=["#2ecc71", "#e74c3c"]
            )
            fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
            fig.update_layout(yaxis=dict(range=[0,1]))
            st.plotly_chart(fig, use_container_width=True)

        # Grad-CAM side by side
        if show_gradcam:
            if not GRADCAM_AVAILABLE:
                st.error("pytorch-grad-cam not installed.")
                st.exception(gradcam_import_error)
            else:
                if hasattr(model, "model"):
                    target_layers = [model.model.layer4[-1]]
                else:
                    target_layers = [model.layer4[-1]]
                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
                rgb_img = np.array(img.resize((224,224))) / 255.0
                grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                st.subheader("Grad-CAM")
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original", width=350)
                with col2:
                    st.image(visualization, caption="Grad-CAM", width=350)

    except Exception as e:
        st.error(f"Failed to process uploaded file: {e}")

# Footer
st.markdown('<p class="footer">Made with ❤️ using Streamlit & PyTorch</p>', unsafe_allow_html=True)
