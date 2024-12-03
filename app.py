import streamlit as st
from PIL import Image
import cv2
import numpy as np

def main():
    st.title("Image Segmentation Web Application")
    st.sidebar.title("Upload & Configure")

    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    method = st.sidebar.selectbox("Choose Segmentation Method", 
                                   ["Thresholding", "Canny Edge Detection", 
                                    "Watershed", "K-Means Clustering"])
    if uploaded_file is not None:
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

if __name__ == "__main__":
    main()
