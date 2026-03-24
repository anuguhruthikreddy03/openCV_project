# Photo Editor using OpenCV and Streamlit

## Project Description
This project is an interactive photo editor built with Streamlit and OpenCV.  
Users can upload an image, apply common enhancement controls, use artistic filters, and download the edited output.

## Tools Used
- Python
- Streamlit
- OpenCV (`opencv-python`)
- NumPy
- Pillow

## Features
- Upload image (`.png`, `.jpg`, `.jpeg`, `.webp`)
- Resize image (width and height sliders)
- Brightness adjustment
- Contrast adjustment
- Grayscale conversion
- Blur effect
- Warm filter
- Portrait-style background blur (face-aware using Haar Cascade)
- Sharpen effect
- Download edited image

### Extra Features
- Edge detection
- Sketch effect
- Image rotation

## Application Flow
Upload Image  
-> Adjust brightness and contrast  
-> Apply filters/effects  
-> View edited image  
-> Download edited image

## Steps to Run the Project
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. (Optional) Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install streamlit opencv-python numpy pillow
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. Open the local URL shown in terminal (usually `http://localhost:8501`).

## Repository Files
- `app.py`
- `README.md`
