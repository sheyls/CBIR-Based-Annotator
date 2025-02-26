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
  - Annotations can be saved and downloaded as a CSV file.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5187d423-ab3c-44db-b64b-70fac5fc5d9f" width="800"/>
</p>

![5](h)

<p align="center">
  <img src="https://github.com/user-attachments/assets/b41a55df-bb91-4d2f-80d5-c002e5a07f3d" width="800"/>
</p>

## Use

```bash
pip install -r requirements.txt
```

```bash
streamlit run streamlit_app.py
```


