import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Thresholding
def apply_threshold(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "global":
        _, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif method == "adaptive":
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    return result

# Canny Edge Detection
def apply_canny(image, low, high):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges

# Watershed Algorithm
def apply_watershed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, markers = cv2.connectedComponents(np.uint8(dist_transform > 0.7 * dist_transform.max()))
    markers = cv2.watershed(image, markers)
    result = np.zeros_like(image)
    result[markers > 0] = [255, 0, 0]  # Mark watershed regions
    return result

# K-Means Clustering
def apply_kmeans(image, k):
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    return segmented_image

# Main App
def main():
    st.title("Image Segmentation Web Application")
    st.sidebar.title("Upload & Configure")

    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    method = st.sidebar.selectbox("Choose Segmentation Method", 
                                   ["Thresholding", "Canny Edge Detection", 
                                    "Watershed", "K-Means Clustering"])
    if uploaded_file is not None:
        try:
            image = np.array(Image.open(uploaded_file))
            st.image(image, caption="Original Image", use_column_width=True)
            
            if method == "Thresholding":
                option = st.sidebar.radio("Thresholding Method", ["Global", "Adaptive"])
                result = apply_threshold(image, option.lower())
            elif method == "Canny Edge Detection":
                low = st.sidebar.slider("Low Threshold", 0, 100, 50)
                high = st.sidebar.slider("High Threshold", 100, 300, 150)
                result = apply_canny(image, low, high)
            elif method == "Watershed":
                result = apply_watershed(image)
            elif method == "K-Means Clustering":
                k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
                result = apply_kmeans(image, k)
            
            st.image(result, caption=f"{method} Result", use_column_width=True)
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
