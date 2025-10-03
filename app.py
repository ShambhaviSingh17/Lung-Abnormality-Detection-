# -*- coding: utf-8 -*-
"""
CliniScan: Merged Chest X-Ray Classification, Training, and Deployment Script.

"""

# ===================================================================
# 0. IMPORTS
# ===================================================================

# Standard library imports
import os
import io
import tempfile
from io import BytesIO

# Third-party imports
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import pydicom
import plotly.express as px
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Optional Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError as e:
    GRADCAM_AVAILABLE = False
    gradcam_import_error = e


# ===================================================================
# SECTION 1: CORE COMPONENTS (MODEL & UTILITIES)
# ===================================================================

# --- 1.1 ResNet Classifier Definition ---
# This class defines the ResNet-18 based classifier. It robustly handles
# different torchvision versions for loading pretrained weights.
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNetClassifier, self).__init__()
        # Use new `weights` argument if available, otherwise fallback to `pretrained`
        try:
            from torchvision.models import ResNet18_Weights
            self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        except ImportError:
            self.model = models.resnet18(pretrained=True)
            
        # Modify the final fully connected layer for the desired number of classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# --- 1.2 DICOM to PIL Utility (for Streamlit app) ---
# Converts in-memory DICOM bytes directly to a PIL Image for processing.
def dicom_bytes_to_pil(dcm_bytes: bytes) -> Image.Image:
    """Convert DICOM bytes to a PIL.Image (RGB)."""
    try:
        ds = pydicom.dcmread(BytesIO(dcm_bytes))
        arr = ds.pixel_array
        # Normalize pixel values to 0–255 and convert to uint8
        arr_norm = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        # Convert to a 3-channel RGB image
        img = Image.fromarray(arr_norm).convert("RGB")
        return img
    except Exception as e:
        raise RuntimeError(f"Failed to parse DICOM: {e}")


# ===================================================================
# SECTION 2: DATASET PREPARATION & MODEL TRAINING (OFFLINE TASKS)
# ===================================================================

# --- 2.1 DICOM to PNG File Conversion Utilities ---
# For batch-converting a DICOM dataset to PNG files on disk.
def dicom_to_png(dicom_path: str, png_path: str) -> bool:
    """Convert a single DICOM file to PNG format."""
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype("uint8")
        cv2.imwrite(png_path, img_uint8)
        return True
    except Exception as e:
        print(f"❌ Failed to convert {dicom_path}: {e}")
        return False

def convert_dataset(dicom_dir: str, output_dir: str, annotations_csv: str):
    """Batch convert DICOM dataset to PNG using an annotations CSV."""
    os.makedirs(output_dir, exist_ok=True)
    annotations = pd.read_csv(annotations_csv)
    print("🚀 Starting DICOM to PNG conversion...")
    for _, row in annotations.iterrows():
        dicom_file = os.path.join(dicom_dir, f"{row['image_id']}.dicom")
        png_file = os.path.join(output_dir, f"{row['image_id']}.png")
        if os.path.exists(dicom_file):
            if dicom_to_png(dicom_file, png_file):
                print(f"✅ Converted: {dicom_file} → {png_file}")
        else:
            print(f"⚠️ Missing file: {dicom_file}")
    print("✅ Conversion complete.")

# --- 2.2 Custom PyTorch Dataset for VinDr-CXR ---
# A custom dataset class to handle the specific format of the VinDr-CXR dataset.
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
            # This part remains for completeness if detection is needed
            boxes, labels = [], []
            for _, row in annotations.iterrows():
                boxes.append([row['x_min'], row['y_min'], row['x_max'], row['y_max']])
                labels.append(self.label_to_id[row['label']])
            target = {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}
            if self.transform:
                image = self.transform(image)
            return image, target


# ===================================================================
# SECTION 3: STREAMLIT APPLICATION (MAIN EXECUTION)
# ===================================================================

# --- 3.1 Model Loading Helper ---
@st.cache_resource
def load_model_from_path(model_path=None, num_classes=2):
    """Loads the model from a file path or falls back to a pretrained default."""
    model = ResNetClassifier(num_classes=num_classes)
    if model_path:
        try:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            st.success("✅ Custom `model.pth` loaded successfully.")
        except Exception as e:
            st.warning(f"Could not load custom model: {e}. Falling back to default.")
    else:
        st.info("No custom model uploaded. Using default pretrained ResNet18.")
    
    model.eval()
    return model

# --- 3.2 Preprocessing Transforms for Inference ---
transform_inference = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3.3 Main Application Logic ---
def main():
    st.set_page_config(page_title="CliniScan — Chest X-Ray AI", layout="wide")

    # Custom styling
    st.markdown("""
        <style>
        .big-title { font-size:36px !important; font-weight:bold; color:#00BFFF; text-align:center; }
        .footer { font-size:14px; text-align:center; color:gray; margin-top:50px; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-title">🩻 CliniScan — Chest X-Ray Classification + Grad-CAM</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>An AI tool to classify chest X-rays and visualize model attention.</p>", unsafe_allow_html=True)
    
    # Sidebar for options
    with st.sidebar:
        st.header("⚙️ Configuration")
        num_classes = st.number_input("Number of classes", value=2, min_value=2, max_value=100)
        upload_model = st.file_uploader("Upload a custom `model.pth`", type=["pt", "pth"])
        st.header("👁️ Visualization")
        show_gradcam = st.checkbox("Enable Grad-CAM visualization", value=True)
        show_probs = st.checkbox("Show class probabilities chart", value=True)

    # Handle uploaded model
    model_path = None
    if upload_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tfile:
            tfile.write(upload_model.getvalue())
            model_path = tfile.name

    model = load_model_from_path(model_path=model_path, num_classes=int(num_classes))

    # File uploader
    uploaded_file = st.file_uploader("📤 Upload a Chest X-ray image", type=["png", "jpg", "jpeg", "dcm"])

    if uploaded_file is None:
        st.info("Please upload an image file (PNG, JPG) or a DICOM file (.dcm).")
    else:
        try:
            # Handle DICOM or standard image formats
            if uploaded_file.name.lower().endswith(".dcm"):
                image = dicom_bytes_to_pil(uploaded_file.getvalue())
            else:
                image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")

            # Preprocess and predict
            input_tensor = transform_inference(image).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                pred_class_idx = torch.argmax(probs).item()

            class_names = [f"Class_{i}" for i in range(int(num_classes))]
            if int(num_classes) == 2:
                class_names = ["Normal", "Pneumonia"] # Example labels
            
            pred_class_name = class_names[pred_class_idx]
            confidence = probs[pred_class_idx].item() * 100

            st.subheader("📊 Prediction Results")
            st.metric(label="Predicted Class", value=pred_class_name, delta=f"{confidence:.2f}% Confidence")

            # Display probability chart
            if show_probs:
                chart_data = pd.DataFrame({"Class": class_names, "Probability": probs.numpy()})
                fig = px.bar(chart_data, x="Class", y="Probability", color="Class", text_auto='.2%')
                fig.update_layout(yaxis=dict(range=[0,1]), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Grad-CAM Visualization
            if show_gradcam:
                if not GRADCAM_AVAILABLE:
                    st.error("`pytorch-grad-cam` is not installed. Cannot generate heatmap.")
                    st.exception(gradcam_import_error)
                else:
                    st.subheader("🔥 Grad-CAM Heatmap")
                    try:
                        target_layers = [model.model.layer4[-1]]
                        cam = GradCAM(model=model, target_layers=target_layers)
                        rgb_img = np.array(image.resize((224, 224))) / 255.0
                        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class_idx)])[0, :]
                        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image, caption="Original Uploaded Image", use_column_width=True)
                        with col2:
                            st.image(visualization, caption=f"Grad-CAM for '{pred_class_name}'", use_column_width=True)
                    except Exception as e:
                        st.error(f"Failed to generate Grad-CAM visualization: {e}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            
    st.markdown('<p class="footer">CliniScan AI Tool | Built with Streamlit & PyTorch</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
