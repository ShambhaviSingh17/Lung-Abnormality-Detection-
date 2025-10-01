import os
import io
import zipfile
import shutil
from typing import List

import cv2
import pydicom
import pandas as pd
import numpy as np
from PIL import Image

#import torch
#import torch.nn as nn
from torchvision import transforms, models

import streamlit as st

# Optional: grad-cam (only used if user checks the option and package installed)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except Exception:
    HAS_GRAD_CAM = False


st.set_page_config(page_title="Cliniscan (DICOM -> PNG & Inference)", layout="wide")


# -----------------------
# Utilities: DICOM -> PNG
# -----------------------
def dicom_to_png(dicom_path: str, png_path: str) -> bool:
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array

        # Normalize to 0-255 and convert to uint8
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype("uint8")

        # If single channel, write as grayscale. If multi-channel, try to save properly.
        cv2.imwrite(png_path, img_uint8)
        return True
    except Exception as e:
        st.error(f"Failed to convert {dicom_path}: {e}")
        return False


def convert_uploaded_dicoms_to_png(uploaded_files: List[io.BytesIO], output_dir: str) -> int:
    """
    Accepts a list of uploaded DICOM files (Streamlit UploadedFile objects or BytesIO)
    and writes PNGs into output_dir. Returns number of successful conversions.
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for uploaded in uploaded_files:
        # uploaded: streamlit UploadedFile (has .name and .getbuffer)
        if hasattr(uploaded, "getbuffer"):
            name = uploaded.name
            data = uploaded.getbuffer()
            # write temporary dicom file then convert
            tmp_dcm = os.path.join(output_dir, f"tmp_{name}")
            with open(tmp_dcm, "wb") as f:
                f.write(data)

            png_name = os.path.splitext(name)[0] + ".png"
            png_path = os.path.join(output_dir, png_name)
            ok = dicom_to_png(tmp_dcm, png_path)
            os.remove(tmp_dcm)
            if ok:
                count += 1
    return count


def extract_zip_to_folder(zip_file_obj, extract_to: str) -> None:
    """Extract an uploaded zip file (UploadedFile or BytesIO) to a folder."""
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_file_obj.getvalue())) as z:
        z.extractall(extract_to)


# -----------------------
# Dataset class (kept for reference / future use)
# -----------------------
import torch
from torch.utils.data import Dataset

class VinDrCXRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, task='classification'):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.task = task

        # ensure 'label' exists
        if 'label' in df.columns:
            self.labels = sorted(df['label'].unique())
        else:
            self.labels = []
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}

        self.grouped_images = self.df.groupby('image_id') if 'image_id' in df.columns else {}
        self.image_ids = list(self.grouped_images.groups.keys()) if self.grouped_images else []

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
            boxes = []
            labels = []
            for _, row in annotations.iterrows():
                boxes.append([row['x_min'], row['y_min'], row['x_max'], row['y_max']])
                labels.append(self.label_to_id[row['label']])
            target = {}
            target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.tensor(labels, dtype=torch.int64)
            if self.transform:
                image = self.transform(image)
            return image, target

# -----------------------
# Model wrapper
# -----------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNetClassifier, self).__init__()
        # Load a ResNet-18. We use weights=None by default (not downloading when running on Streamlit)
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_model_from_file(path: str, num_classes: int = 2, device="cpu"):
    model = ResNetClassifier(num_classes)
    try:
        state = torch.load(path, map_location=device)
        # allow both state dict and whole model saved
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif isinstance(state, dict):
            model.load_state_dict(state)
        else:
            model = state
        model.eval()
        st.success("Model loaded successfully.")
    except Exception as e:
        st.warning(f"Could not load model.pth: {e}\nUsing untrained model instead.")
        model.eval()
    return model


# -----------------------
# Transforms
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# -----------------------
# Streamlit layout
# -----------------------
st.title("Cliniscan — DICOM → PNG & Image Inference")
st.markdown("Upload DICOM files, convert to PNGs, optionally upload a `model.pth` for inference, or visualize Grad-CAM.")

with st.expander("1) Convert DICOM files to PNG"):
    st.write("Upload one or more DICOM files or a ZIP containing DICOMs.")
    uploaded_dcm_files = st.file_uploader("Upload DICOM files (.dcm, .dicom) (multiple)", type=["dcm", "dicom"], accept_multiple_files=True)
    uploaded_zip = st.file_uploader("Or upload a ZIP containing DICOMs", type=["zip"])

    out_dir = st.text_input("Output folder (in app filesystem)", value="png_output")
    if st.button("Convert uploaded DICOMs"):
        if uploaded_dcm_files:
            n = convert_uploaded_dicoms_to_png(uploaded_dcm_files, out_dir)
            st.success(f"Converted {n} files → saved to `{out_dir}`")
            # show preview of converted files
            files = [f for f in os.listdir(out_dir) if f.lower().endswith(".png")]
            if files:
                st.image([os.path.join(out_dir, f) for f in files[:5]], width=200)
        else:
            st.warning("No DICOM files uploaded. If you uploaded a ZIP, use the 'Extract ZIP' button below.")

    if uploaded_zip:
        extract_to = st.text_input("Extract ZIP to folder", value="dataset_from_zip")
        if st.button("Extract ZIP"):
            try:
                extract_zip_to_folder(uploaded_zip, extract_to)
                st.success(f"ZIP extracted to `{extract_to}`")
                # Optionally convert all .dcm files under extract_to
                if st.button("Convert all extracted .dcm to PNG"):
                    dicom_paths = []
                    for root, _, files in os.walk(extract_to):
                        for f in files:
                            if f.lower().endswith((".dcm", ".dicom")):
                                dicom_paths.append(os.path.join(root, f))
                    # Convert
                    os.makedirs(out_dir, exist_ok=True)
                    converted = 0
                    for dpath in dicom_paths:
                        png_name = os.path.splitext(os.path.basename(dpath))[0] + ".png"
                        png_path = os.path.join(out_dir, png_name)
                        if dicom_to_png(dpath, png_path):
                            converted += 1
                    st.success(f"Converted {converted} DICOMs from ZIP → saved to `{out_dir}`")
            except Exception as e:
                st.error(f"Could not extract ZIP: {e}")

st.write("---")
with st.expander("2) Load model (optional)"):
    st.write("Upload a PyTorch `model.pth` if you want to run inference with your trained model. Otherwise a dummy (untrained) ResNet-18 will be used.")
    uploaded_model = st.file_uploader("Upload model.pth (optional)", type=["pth", "pt"])
    model_obj = None
    if uploaded_model:
        # save temporarily
        tmp_model_path = "model_uploaded.pth"
        with open(tmp_model_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        # Ask for number of classes
        num_classes = st.number_input("Number of classes in your model", min_value=1, value=2)
        model_obj = load_model_from_file(tmp_model_path, num_classes=num_classes, device="cpu")
    else:
        # create a default untrained model (num_classes=2)
        if st.button("Load default ResNet-18 (untrained)"):
            model_obj = load_model_from_file("", num_classes=2, device="cpu")  # will warn and return untrained model

st.write("---")
with st.expander("3) Run inference on an image"):
    uploaded_img = st.file_uploader("Upload an image (PNG/JPG) for prediction", type=["png", "jpg", "jpeg"])
    run_gradcam = st.checkbox("Produce Grad-CAM visualization (requires pytorch-grad-cam)", value=False)
    if run_gradcam and not HAS_GRAD_CAM:
        st.warning("Grad-CAM package not installed. Add `pytorch-grad-cam` to requirements to enable this.")

    if uploaded_img:
        # display
        image_pil = Image.open(uploaded_img).convert("RGB")
        st.image(image_pil, caption="Uploaded image", use_column_width=False, width=350)

        # Ensure model_obj exists
        if model_obj is None:
            st.info("No model loaded - using an untrained ResNet-18 with 2 classes for demonstration.")
            model_obj = load_model_from_file("", num_classes=2, device="cpu")

        # Prepare input
        input_tensor = transform(image_pil).unsqueeze(0)  # shape (1,3,224,224)
        with torch.no_grad():
            outputs = model_obj(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_class = torch.argmax(probs).item()

        # display prediction
        class_names = [f"Class_{i}" for i in range(probs.shape[0])]
        st.write(f"### Prediction: **{class_names[pred_class]}**")
        st.write(f"Confidence: {probs[pred_class].item()*100:.2f}%")
        # bar chart
        st.bar_chart({class_names[i]: float(probs[i].item()) for i in range(len(class_names))})

        # Grad-CAM visualization if requested and available
        if run_gradcam and HAS_GRAD_CAM:
            try:
                rgb_img = np.array(image_pil.resize((224, 224))) / 255.0
                target_layers = [model_obj.model.layer4[-1]] if hasattr(model_obj, "model") else [list(model_obj.children())[-1]]
                cam = GradCAM(model=model_obj, target_layers=target_layers, use_cuda=False)
                grayscale_cam = cam(input_tensor=input_tensor)[0]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                st.image(visualization, caption="Grad-CAM", use_column_width=False, width=350)
            except Exception as e:
                st.error(f"Could not run Grad-CAM: {e}")

st.write("---")
st.markdown("### Notes & Next steps")
st.markdown("""
- This app is **not** running any heavy training on Streamlit. Training models should be done offline or on proper compute (Colab/VM) and the trained `model.pth` uploaded for inference.
- To deploy: put this `app.py` and a `requirements.txt` into your GitHub repo and connect the repo on Streamlit Cloud.
- If you need YOLO/Ultralytics training or large dataset extraction, do that offline/Colab, then push resulting artifacts/models to GitHub or cloud storage and load them here.
""")

