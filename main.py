import os
import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image
import streamlit as st
import pandas as pd

# ============================
# Feature Extraction
# ============================
class FeatureExtractor:
    def __init__(self, bins=(8, 8, 8)):
        self.bins = bins

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))
        
        return image

    def extract_features(self, image):
        # Color Histogram
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # HOG Features
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hog_features = hog(
            gray_image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True,
            visualize=False,
            feature_vector=True
        )
        features = np.concatenate([hist, hog_features])
        return features

# ============================
# CBIR System
# ============================
class CBIRSystem:
    def __init__(self, dataset_folder, extractor):
        self.dataset_folder = dataset_folder
        self.extractor = extractor
        self.image_features = {}  # {image_path: feature_vector}
        self.load_dataset()

    def load_dataset(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        for filename in os.listdir(self.dataset_folder):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(self.dataset_folder, filename)
                try:
                    image = self.extractor.preprocess_image(image_path)
                    features = self.extractor.extract_features(image)
                    self.image_features[image_path] = features
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    def retrieve_similar_images(self, query_image_path, top_k=None):
        query_image = self.extractor.preprocess_image(query_image_path)
        query_features = self.extractor.extract_features(query_image)
        results = []
        for image_path, features in self.image_features.items():
            distance = np.linalg.norm(query_features - features)
            results.append((image_path, distance))
        results.sort(key=lambda x: x[1])
        if top_k is None:
            return results
        else:
            return results[:top_k]

