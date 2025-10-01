# CliniScan: AI Lung Abnormality Detection

CliniScan is an **AI-powered system** that automatically detects and localizes lung abnormalities from chest X-ray images.  
The script uses a **deep learning model trained on the VinDr-CXR dataset** to identify findings like *opacities* and *fibrosis*, aiming to assist radiologists in their diagnostic workflow.

---

##  Prerequisites

This project requires **Python** and several key libraries to function correctly.  
You can install all necessary modules using the `requirements.txt` file.

**Key modules include:**
- streamlit  
- torch & torchvision  
- pydicom  
- opencv-python-headless  
- pandas  

To install them, run the following command in your terminal:

```bash
pip install -r requirements.txt
