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

## Use

```bash
streamlit run streamlit_app.py
```


