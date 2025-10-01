**CliniScan: AI Lung Abnormality Detection**
This project uses an AI model to automatically detect and highlight potential lung abnormalities in chest X-ray images. It's designed to assist radiologists by quickly identifying areas of concern.

**How It Works**
Data: The model is trained on the VinDr-CXR dataset, which contains thousands of annotated chest X-ray images.

Preprocessing: Images are cleaned, resized, and prepared for the model.

AI Model: A deep learning model (like YOLOv8) is trained to recognize and draw bounding boxes around abnormalities.

Output: The system outputs the X-ray image with potential issues highlighted for review.

**Technology Used**
Frameworks: PyTorch / TensorFlow

Models: YOLOv8, EfficientNet, ResNet

Key Libraries: pydicom, pandas, OpenCV

App (Optional): Streamlit

**Getting Started**
Clone the repository:

git clone <your-repository-url>
cd <repository-name>

Install dependencies:

pip install -r requirements.txt

Download the Dataset:

Get the VinDr-CXR dataset and place it in the designated data folder.

Run the application:

streamlit run app.py
