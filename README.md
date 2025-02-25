# Flower CBIR and Annotation System

This repository implements a Content-Based Image Retrieval (CBIR) system and annotation tool for flower images. The system is designed to streamline the image annotation process by automatically retrieving visually similar images based on efficient feature descriptors.

## Overview

The project consists of two main components:

- **Feature Extraction & CBIR System:**  
  - **Preprocessing:** Loads images, converts color spaces, removes the background using a simple corner-based estimation, and resizes images to a fixed dimension (1080x1080).
  - **Descriptors:**  
    - *Color Histogram:* Fast and efficient, capturing the global color distribution – particularly useful for flower images where color is a crucial feature.
    - *HOG (Histogram of Oriented Gradients):* Captures structural and edge information, complementing the color histogram.
  - **Similarity Metrics:** Provides various metrics (Euclidean, cosine, Manhattan, histogram intersection, chi-square) to compute distances between feature vectors.

- **Streamlit-Based Annotator Interface:**  
  - Allows users to upload a query image and specify a reference flower name.
  - Retrieves similar images from a specified dataset.
  - Enables interactive annotation with feedback buttons to mark if retrieved images match the reference flower.
  - Optionally, annotations can be saved and downloaded as a CSV file.
    
<p align="center">
  <img src="https://github.com/user-attachments/assets/05a99749-525a-4a88-bd6c-02a63f79e8d1" width="800"/>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/1af3de8e-c76c-40d8-8071-95c5208bb003" width="800"/>
</p>

## Use

```bash
pip install -r requirements.txt
```

```bash
streamlit run streamlit_app.py
```


