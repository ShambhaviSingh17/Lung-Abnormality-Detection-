import os
import cv2
import pydicom
import pandas as pd
import streamlit as st

def dicom_to_png(dicom_path: str, png_path: str) -> bool:
    try:
        ds = pydicom.dcmread(dicom_path)
        img = ds.pixel_array

        # Normalize image to 0-255
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype("uint8")

        cv2.imwrite(png_path, img_uint8)
        return True
    except Exception as e:
        st.error(f"Failed to convert {dicom_path}: {e}")
        return False


def convert_dataset(dicom_dir: str, output_dir: str, annotations_csv: str):
    os.makedirs(output_dir, exist_ok=True)

    annotations = pd.read_csv(annotations_csv)

    for _, row in annotations.iterrows():
        dicom_file = os.path.join(dicom_dir, f"{row['image_id']}.dicom")
        png_file = os.path.join(output_dir, f"{row['image_id']}.png")

        if os.path.exists(dicom_file):
            success = dicom_to_png(dicom_file, png_file)
            if success:
                st.success(f"✅ Converted: {dicom_file} → {png_file}")
        else:
            st.warning(f"⚠️ Missing file: {dicom_file}")


# -----------------------
# Streamlit UI
# -----------------------
st.title("DICOM to PNG Converter")

st.write("Upload your **annotations.csv** and DICOM files to convert them into PNG format.")

uploaded_csv = st.file_uploader("Upload Annotations CSV", type="csv")
uploaded_dicoms = st.file_uploader("Upload DICOM Files", type=["dcm", "dicom"], accept_multiple_files=True)

if uploaded_csv and uploaded_dicoms:
    # Save uploaded files temporarily
    dicom_dir = "dicom_input"
    output_dir = "png_output"
    os.makedirs(dicom_dir, exist_ok=True)

    # Save CSV
    annotations_csv = "annotations.csv"
    with open(annotations_csv, "wb") as f:
        f.write(uploaded_csv.getbuffer())

    # Save all dicom files
    for dicom_file in uploaded_dicoms:
        with open(os.path.join(dicom_dir, dicom_file.name), "wb") as f:
            f.write(dicom_file.getbuffer())

    st.info("Files uploaded successfully. Click 'Start Conversion' to begin.")

    if st.button("Start Conversion"):
        convert_dataset(dicom_dir, output_dir, annotations_csv)
        st.success(f"🎉 Conversion complete! Files saved in `{output_dir}/`")
