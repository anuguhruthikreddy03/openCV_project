import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="OpenCV Photo Editor", layout="wide")
st.title("Photo Editor using OpenCV and Streamlit")
st.write("Upload an image, tune adjustments, apply effects, and download your final result.")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def adjust_brightness_contrast(img: np.ndarray, brightness: int, contrast: int) -> np.ndarray:
    alpha = 1.0 + (contrast / 100.0)
    beta = brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def apply_warm_filter(img: np.ndarray, intensity: float) -> np.ndarray:
    b, g, r = cv2.split(img.astype(np.float32))
    # Warm tone: boost red and slightly reduce blue.
    r = np.clip(r * (1.0 + 0.25 * intensity), 0, 255)
    b = np.clip(b * (1.0 - 0.20 * intensity), 0, 255)
    warmed = cv2.merge([b, g, r]).astype(np.uint8)
    return warmed


def portrait_background_blur(img: np.ndarray, blur_strength: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    k = max(1, blur_strength)
    k = k + 1 if k % 2 == 0 else k
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    if len(faces) == 0:
        return blurred

    mask = np.zeros(gray.shape, dtype=np.uint8)
    h_img, w_img = gray.shape

    for (x, y, w, h) in faces:
        cx = x + w // 2
        cy = y + h // 2
        rx = int(w * 0.95)
        ry = int(h * 1.25)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask_f = (mask / 255.0)[..., np.newaxis]

    output = (img.astype(np.float32) * mask_f) + (blurred.astype(np.float32) * (1.0 - mask_f))
    return np.clip(output, 0, 255).astype(np.uint8)


def sharpen_image(img: np.ndarray, amount: float) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(img, -1, kernel)
    return cv2.addWeighted(img, 1.0 - amount, sharp, amount, 0)


def edge_effect(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def sketch_effect(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)
    original_bgr = pil_to_bgr(pil_img)
    edited = original_bgr.copy()

    st.sidebar.header("Adjustments")
    target_width = st.sidebar.slider("Resize width (px)", 100, 2000, int(original_bgr.shape[1]), 10)
    target_height = st.sidebar.slider("Resize height (px)", 100, 2000, int(original_bgr.shape[0]), 10)

    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", -100, 100, 0)

    grayscale = st.sidebar.checkbox("Convert to Grayscale")
    blur_strength = st.sidebar.slider("Blur strength", 0, 31, 0, 2)
    warm_intensity = st.sidebar.slider("Warm filter intensity", 0.0, 1.0, 0.0, 0.05)
    portrait_blur = st.sidebar.checkbox("Portrait-style background blur")
    portrait_strength = st.sidebar.slider("Portrait blur strength", 1, 35, 17, 2)
    sharpen_amount = st.sidebar.slider("Sharpen amount", 0.0, 1.0, 0.0, 0.05)

    st.sidebar.header("Extra Features")
    rotation_angle = st.sidebar.slider("Rotate (degrees)", -180, 180, 0, 1)
    edge_toggle = st.sidebar.checkbox("Edge detection")
    sketch_toggle = st.sidebar.checkbox("Sketch effect")

    edited = cv2.resize(edited, (target_width, target_height), interpolation=cv2.INTER_AREA)
    edited = adjust_brightness_contrast(edited, brightness=brightness, contrast=contrast)

    if grayscale:
        gray = cv2.cvtColor(edited, cv2.COLOR_BGR2GRAY)
        edited = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if blur_strength > 0:
        k = blur_strength + 1 if blur_strength % 2 == 0 else blur_strength
        edited = cv2.GaussianBlur(edited, (k, k), 0)

    if warm_intensity > 0:
        edited = apply_warm_filter(edited, warm_intensity)

    if portrait_blur:
        edited = portrait_background_blur(edited, portrait_strength)

    if sharpen_amount > 0:
        edited = sharpen_image(edited, sharpen_amount)

    if rotation_angle != 0:
        edited = rotate_image(edited, rotation_angle)

    if edge_toggle:
        edited = edge_effect(edited)

    if sketch_toggle:
        edited = sketch_effect(edited)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(bgr_to_rgb(original_bgr), use_container_width=True)
    with col2:
        st.subheader("Edited Image")
        st.image(bgr_to_rgb(edited), use_container_width=True)

    out_rgb = bgr_to_rgb(edited)
    out_pil = Image.fromarray(out_rgb)
    buffer = io.BytesIO()
    out_pil.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="Download Edited Image",
        data=buffer,
        file_name="edited_image.png",
        mime="image/png",
    )
else:
    st.info("Please upload an image to start editing.")
