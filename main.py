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
    def __init__(self, bins=4, mode="hist"):
        """
        bins: Number of bins for the color histogram for each channel.
        """
        assert mode in ["hist", "hog", "both"]
        self.mode = mode
        self.bins = bins
        self.output_info = None


    def preprocess_image(self, image_path):
        """
        Carga la imagen, la convierte de BGR a RGB, elimina el fondo basado en el color de fondo
        calculado a partir de las esquinas, y la redimensiona.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Obtener dimensiones de la imagen
        h, w, _ = image.shape

        # Tomar muestras de colores de fondo de las esquinas (superior izquierda, superior derecha,
        # inferior izquierda, inferior derecha)
        corner_pixels = np.concatenate([
            image[0:10, 0:10],    # Esquina superior izquierda
            image[0:10, -10:],    # Esquina superior derecha
            image[-10:, 0:10],    # Esquina inferior izquierda
            image[-10:, -10:]     # Esquina inferior derecha
        ])

        # Aplanar para tener una lista de píxeles con forma (N, 3)
        corner_pixels = corner_pixels.reshape(-1, 3)
        
        # Calcular el color de fondo (mediana de todos los píxeles de las esquinas)
        bg_color = np.median(corner_pixels, axis=0)
        bg_color = np.array(bg_color, dtype=np.uint8)

        # Crear un array del mismo tamaño que la imagen, con el color de fondo
        bg_color_full = np.full_like(image, bg_color)

        # Definir un umbral para la similitud al fondo
        threshold = 40  # Ajusta según necesites

        # Calcular la diferencia euclidiana entre cada píxel y el color de fondo
        diff = np.linalg.norm(image - bg_color_full, axis=2)

        # Crear una máscara: píxeles con diferencia mayor al umbral se consideran primer plano
        mask = (diff > threshold).astype(np.uint8) * 255  # Formato 0-255

        # Aplicar la máscara para extraer el primer plano
        foreground = cv2.bitwise_and(image, image, mask=mask)

        # Redimensionar la imagen resultante
        fixed_size = (1080, 1080)
        foreground = cv2.resize(foreground, fixed_size)
        return foreground


    def extract_features(self, image):
        """
        Extracts two types of features:
         - Color histogram (RGB)
         - HOG features (from grayscale image)
        Then concatenates both feature vectors.
        """
        if self.mode == "hist" or self.mode == "both":
            # --- Color Histogram ---
            hist = cv2.calcHist([image], [0, 1, 2], None, (self.bins,self.bins,self.bins), [0, 256, 0, 256, 0, 256])
            hist_features = cv2.normalize(hist, hist).flatten()
        else:
            hist_features = np.zeros(shape=(0,))

        if self.mode != "hist":
            # --- HOG Features ---
            #fixed_size = (1080, 1080)
            #image = cv2.resize(image, fixed_size)

            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            w, h = gray_image.shape
            pixels_per_cell = (w//self.bins, h//self.bins)

            hog_features = hog(
                gray_image,
                orientations=9,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                transform_sqrt=True,
                visualize=False,
                feature_vector=True
            )
        else:
            hog_features = np.zeros(shape=(0,))

        print(hog_features.shape, hist_features.shape)

        features = np.concatenate([hist_features, hog_features])
        if self.output_info is None:
            self.output_info = (hist_features.size, hog_features.size)
        return features

# ============================
# CBIR System
# ============================
class CBIRSystem:
    def __init__(self, dataset_folder, extractor, limit=None, metric="manhattan", use_weights=False):
        """
        dataset_folder: Path to the folder containing the flower images.
        extractor: An instance of FeatureExtractor to process images.
        """
        self.dataset_folder = dataset_folder
        self.extractor = extractor
        self.image_features = {}  # Dictionary: {image_path: feature_vector}
        self.limit = limit
        self.load_dataset()
        self.use_weights = use_weights
        self.weights = None
        self.lr = 0.001
        self.beta = 0.5
        self.grad = None
        self.metric = metric
        if self.use_weights and self.metric not in ["manhattan", "euclidean"]:
            raise ValueError(f"Metric must be either 'manhattan' or 'euclidean' if finetuning is enabled")


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

    def retrieve_similar_images(self, query_image_path, top_k=10):
        """
        Given a query image, extract its features and return the top_k most similar images.
        Similarity is measured via the Euclidean distance.
        """
        query_image = self.extractor.preprocess_image(query_image_path)
        query_features = self.extractor.extract_features(query_image)
        if self.use_weights is True and self.weights is None:
            self.weights = np.ones(query_features.shape)
            a,b = self.extractor.output_info
            self.weights[0:a] = 0.7 / a
            self.weights[a+1:a+b] = 0.3 / b

        results = []
        for image_path, features in self.image_features.items():
            # Euclidean distance: lower distance means higher similarit
            if self.metric == "euclidean":
                if self.weights is not None:
                    # Compute weighted Euclidean distance as sqrt(sum_i w_i * (x_i-y_i)^2)
                    diff = query_features - features
                    distance = np.sqrt(np.sum(self.weights * (diff ** 2)))
                else:
                    distance = euclidean(query_features, features)

            elif self.metric == "manhattan":
                if self.weights is not None:
                    # Weighted Manhattan distance: sum_i (w_i * |x_i-y_i|)
                    distance = np.sum(self.weights * np.abs(query_features - features))
                else:
                    distance = cityblock(query_features, features)

            elif self.metric == "cosine":
                # Using SciPy’s cosine distance (which returns 1 - cosine similarity)
                distance = cosine(query_features, features)

            elif self.metric == "histogram intersection":
                distance = 1 - cv2.compareHist(query_features.astype(np.float32),
                                               features.astype(np.float32),
                                               cv2.HISTCMP_INTERSECT)

            elif self.metric == "chi-square":
                distance = cv2.compareHist(query_features.astype(np.float32),
                                           features.astype(np.float32),
                                           cv2.HISTCMP_CHISQR)

            else:
                raise ValueError("Invalid metric")

            results.append((image_path, distance))
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def finetune(self, query_img, correct_imgs, incorrect_imgs):
        """
        Fine-tunes the weight vector by performing a gradient update based on a triplet loss.

        Parameters:
          query_img      : The query image.
          correct_imgs   : A list of images that are similar/positive examples.
          incorrect_imgs : A list of images that are dissimilar/negative examples.

        Assumptions:
          - self.extract_features(image) returns a feature vector for an image.
          - self.weights is a numpy array of shape (n_features,).
          - self.metric is either "euclidean" or "manhattan".
          - Optionally, self.lr (learning rate) and self.margin (margin for the loss) are defined.
        """
        # Set hyperparameters (or use defaults)
        epsilon = 1e-8  # small constant to avoid division by zero

        # Extract features for the query, correct, and incorrect images.
        query_image = self.extractor.preprocess_image(query_img)
        query_features = self.extractor.extract_features(query_image)

        correct_features = [self.extractor.extract_features(self.extractor.preprocess_image(img)) for img in correct_imgs]
        incorrect_features = [self.extractor.extract_features(self.extractor.preprocess_image(img)) for img in incorrect_imgs]

        # Initialize gradient accumulator for the weights.
        grad = np.zeros_like(self.weights)

        # Iterate over each positive and negative pair.
        for pos_feat in correct_features:
            # Compute distance and its gradient contribution for the positive (correct) example.
            if self.metric == "euclidean":
                diff_pos = query_features - pos_feat
                # Weighted Euclidean distance: sqrt(sum_i w_i * (diff_i)^2)
                pos_dist = np.sqrt(np.sum(self.weights * (diff_pos ** 2))) + epsilon
                grad += diff_pos / pos_dist
            elif self.metric == "manhattan":
                diff_pos = np.abs(query_features - pos_feat)
                # Weighted Manhattan distance: sum_i w_i * |diff_i|
                grad += diff_pos
            else:
                raise ValueError("Unsupported metric: " + self.metric)

        for neg_feat in incorrect_features:
            # Compute distance and its gradient contribution for the negative (incorrect) example.
            if self.metric == "euclidean":
                diff_neg = query_features - neg_feat
                neg_dist = np.sqrt(np.sum(self.weights * (diff_neg ** 2))) + epsilon
                grad -= diff_neg / neg_dist
            elif self.metric == "manhattan":
                diff_neg = np.abs(query_features - neg_feat)
                grad -= diff_neg
            else:
                raise ValueError("Unsupported metric: " + self.metric)

        # Update the weights using a simple gradient descent step.
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.beta * self.grad + (1 - self.beta) * grad
        self.weights -= self.lr * self.grad
        print(self.lr * self.grad)
        self.weights = np.maximum(self.weights, epsilon)


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
            st.write("Finetuning...")

            correct = [path for path, state in st.session_state.feedback.items() if state == "correct"]
            incorrect = [path for path, state in st.session_state.feedback.items() if state == "incorrect"]

            cbir_system.finetune(query_image_path, correct, incorrect)
            st.write("Finetuning finished.")

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

    for k in [8]:
        for mode in ["both"]:
            extractor = FeatureExtractor(bins=k, mode=mode)
            for metric in ["euclidean", "manhattan"]:
                cbir_system = CBIRSystem("dataset/", extractor, metric=metric, use_weights=True, limit=100)
                for it in range(19):
                    results = cbir_system.retrieve_similar_images(query_path, top_k=6)
                    plot_images(results, title=f"CBIR Results - Bins: {k}, Mode: {mode}, Metric: {metric}")
                    cbir_system.finetune(query_path, [results[0][0]], [])
                    print("Finished")


if __name__ == "__main__":
    debug()
    # main()
