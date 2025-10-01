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
```
pip install -r requirements.txt 
```
## How to run the script

1-Clone the repository to your local machine:
```
git clone https://github.com/your-username/CliniScan.git
cd CliniScan
```

2-Install the prerequisites:
```
pip install -r requirements.txt
```

3-Run the Streamlit web application:
```
streamlit run app.py
```

## Demo

Try it here: [CliniScan Live Demo](https://kvekvdmryl6fg9wgsb42fg.streamlit.app/)

