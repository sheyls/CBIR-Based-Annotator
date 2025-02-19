import os
import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image
import streamlit as st

# ============================
# Feature Extraction
# ============================
class FeatureExtractor:
    def __init__(self, bins=(8, 8, 8)):
        """
        bins: Number of bins for the color histogram for each channel.
        """
        self.bins = bins

    def preprocess_image(self, image_path):
        """
        Loads the image, converts from BGR to RGB, and (optionally) resizes it.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Optionally, resize the image (e.g., image = cv2.resize(image, (256, 256)))
        return image

    def extract_features(self, image):
        """
        Extracts two types of features:
         - Color histogram (RGB)
         - HOG features (from grayscale image)
        Then concatenates both feature vectors.
        """
        # --- Color Histogram ---
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        # --- HOG Features ---
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

        # --- Concatenate Features ---
        features = np.concatenate([hist, hog_features])
        return features

# ============================
# CBIR System
# ============================
class CBIRSystem:
    def __init__(self, dataset_folder, extractor):
        """
        dataset_folder: Path to the folder containing the flower images.
        extractor: An instance of FeatureExtractor to process images.
        """
        self.dataset_folder = dataset_folder
        self.extractor = extractor
        self.image_features = {}  # Dictionary: {image_path: feature_vector}
        self.load_dataset()

    def load_dataset(self):
        """
        Iterates over the dataset folder, processes each image, and extracts its features.
        """
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

    def retrieve_similar_images(self, query_image_path, top_k=10):
        """
        Given a query image, extract its features and return the top_k most similar images.
        Similarity is measured via the Euclidean distance.
        """
        query_image = self.extractor.preprocess_image(query_image_path)
        query_features = self.extractor.extract_features(query_image)

        results = []
        for image_path, features in self.image_features.items():
            # Euclidean distance: lower distance means higher similarity
            distance = np.linalg.norm(query_features - features)
            results.append((image_path, distance))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

# ============================
# Streamlit Interface for Feedback
# ============================
def main():
    st.title("Flower CBIR System with Streamlit")

    # Sidebar: set dataset folder, query image, and parameters
    st.sidebar.header("Input Parameters")
    dataset_folder = st.sidebar.text_input("Dataset Folder", "path/to/your/flowers_dataset")
    query_image_path = st.sidebar.text_input("Query Image Path", "path/to/your/query_flower.jpg")
    top_k = st.sidebar.number_input("Top K Images", min_value=1, value=10, step=1)

    # Button to run the retrieval
    if st.sidebar.button("Run Retrieval"):
        try:
            extractor = FeatureExtractor(bins=(8, 8, 8))
            cbir_system = CBIRSystem(dataset_folder, extractor)
            results = cbir_system.retrieve_similar_images(query_image_path, top_k=top_k)
            st.session_state.results = results
            st.session_state.current_index = 0
            st.session_state.feedback = {}
            st.success("Retrieval completed!")
        except Exception as e:
            st.error(f"Error during retrieval: {e}")

    # Check if results exist in session_state
    if "results" in st.session_state:
        current_index = st.session_state.current_index
        results = st.session_state.results

        if current_index < len(results):
            image_path, score = results[current_index]
            st.subheader(f"Image {current_index + 1}/{len(results)}")
            st.write(f"Similarity Score: {score:.2f}")
            try:
                pil_image = Image.open(image_path)
                st.image(pil_image, width=400)
            except Exception as e:
                st.error(f"Error loading image {image_path}: {e}")

            # Two columns for the feedback buttons
            col1, col2 = st.columns(2)
            if col1.button("Yes, it's the flower"):
                st.session_state.feedback[image_path] = "correct"
                st.session_state.current_index += 1
                st.experimental_rerun()
            if col2.button("No, it's not"):
                st.session_state.feedback[image_path] = "incorrect"
                st.session_state.current_index += 1
                st.experimental_rerun()
        else:
            st.success("No more images to review.")
            st.write("Feedback:")
            st.write(st.session_state.feedback)
            # Optionally, allow restarting the process
            if st.button("Restart"):
                for key in ["results", "current_index", "feedback"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.experimental_rerun()

if __name__ == "__main__":
    main()
