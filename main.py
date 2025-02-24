import os
import cv2
import numpy as np
from skimage.feature import hog
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import euclidean, cosine, cityblock
import math


# ============================
# Feature Extraction
# ============================
class FeatureExtractor:
    def __init__(self, bins=8, mode="hog"):
        """
        bins: Number of bins for the color histogram for each channel.
        """
        assert mode in ["hist", "hog"]
        self.mode = mode
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
        if self.mode == "hist":
            # --- Color Histogram ---
            hist = cv2.calcHist([image], [0, 1, 2], None, (self.bins,self.bins,self.bins), [0, 256, 0, 256, 0, 256])
            features = cv2.normalize(hist, hist).flatten()
        else:
            # --- HOG Features ---
            fixed_size = (1080, 1080)
            image = cv2.resize(image, fixed_size)

            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            w, h = gray_image.shape
            pixels_per_cell = (w//self.bins, h//self.bins)

            features = hog(
                gray_image,
                orientations=9,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                transform_sqrt=True,
                visualize=False,
                feature_vector=True
            )

        return features

# ============================
# CBIR System
# ============================
class CBIRSystem:
    def __init__(self, dataset_folder, extractor, limit=None):
        """
        dataset_folder: Path to the folder containing the flower images.
        extractor: An instance of FeatureExtractor to process images.
        """
        self.dataset_folder = dataset_folder
        self.extractor = extractor
        self.image_features = {}  # Dictionary: {image_path: feature_vector}
        self.limit = limit
        self.load_dataset()


    def load_dataset(self):
        """
        Iterates over the dataset folder, processes each image, and extracts its features.
        """
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_counter = 0

        file_list = os.listdir(self.dataset_folder)
        if self.limit is not None:
            file_list = file_list[:self.limit]

        with tqdm(total=len(file_list), desc="Processing Images", unit="img") as pbar:
            for filename in file_list:
                if filename.lower().endswith(valid_extensions):
                    image_path = os.path.join(self.dataset_folder, filename)
                    try:
                        image = self.extractor.preprocess_image(image_path)
                        features = self.extractor.extract_features(image)
                        # print(features.shape)
                        self.image_features[image_path] = features
                    except Exception as e:
                        print(f"\nError processing {image_path}: {e}")

                    image_counter += 1
                pbar.update(1)  # Update progress bar

    def retrieve_similar_images(self, query_image_path, top_k=10, metric="euclidean"):
        """
        Given a query image, extract its features and return the top_k most similar images.
        Similarity is measured via the Euclidean distance.
        """
        query_image = self.extractor.preprocess_image(query_image_path)
        query_features = self.extractor.extract_features(query_image)

        results = []
        for image_path, features in self.image_features.items():
            # Euclidean distance: lower distance means higher similarit
            if metric == "euclidean":
                # Using SciPy’s Euclidean distance implementation
                distance = euclidean(query_features, features)

            elif metric == "cosine":
                # Using SciPy’s cosine distance (which returns 1 - cosine similarity)
                distance = cosine(query_features, features)

            elif metric == "manhattan":
                # Using SciPy’s Manhattan (cityblock) distance implementation
                distance = cityblock(query_features, features)

            elif metric == "histogram intersection":
                distance = 1 - cv2.compareHist(query_features.astype(np.float32),
                                               features.astype(np.float32),
                                               cv2.HISTCMP_INTERSECT)

            elif metric == "chi-square":
                distance = cv2.compareHist(query_features.astype(np.float32),
                                           features.astype(np.float32),
                                           cv2.HISTCMP_CHISQR)

            else:
                raise ValueError("Invalid metric")

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
    dataset_folder = st.sidebar.text_input("Dataset Folder", "dataset/")
    query_image_path = st.sidebar.text_input("Query Image Path", "query_examples/6-16-526503800.jpg")
    top_k = st.sidebar.number_input("Top K Images", min_value=1, value=10, step=1)

    # Button to run the retrieval
    if st.sidebar.button("Run Retrieval"):
        try:
            extractor = FeatureExtractor(bins=32)
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


def plot_images(image_list, title):
    n = len(image_list)
    # Compute the grid size (rows x cols)
    rows = int(math.sqrt(n))
    cols = int(math.ceil(n / rows))

    # Create subplots with the computed grid size
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Flatten axes array (in case it's 2D)
    if n > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot each image in the mosaic
    for i, ax in enumerate(axes):
        if i < n:
            img_path, _ = image_list[i]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(img_path, fontsize=8)  # You can customize font size
        else:
            # Hide any unused subplots
            ax.axis("off")

    # Set the overall title using suptitle
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def debug(query_path="query_examples/6-16-526503800.jpg"):
    for k in [4, 8, 16, 32, 64]:
        for mode in ["hist", "hog"]:
            for metric in ["euclidean", "cosine", "manhattan", "chi-square", "histogram intersection"]:
                extractor = FeatureExtractor(bins=k, mode=mode)
                cbir_system = CBIRSystem("dataset/", extractor, limit=1000)
                results = cbir_system.retrieve_similar_images(query_path, top_k=10, metric=metric)
                plot_images(results, title=f"CBIR Results - Bins: {k}, Mode: {mode}, Metric: {metric}")
                print("Finished")


if __name__ == "__main__":
    debug()
    # main()
