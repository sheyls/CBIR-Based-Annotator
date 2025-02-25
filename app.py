import pandas as pd
import streamlit as st
from PIL import Image
from main import FeatureExtractor, CBIRSystem

# ============================
# Streamlit Annotator Interface
# ============================
def main():
    st.title("Flower CBIR Annotator")

    if "global_annotations" not in st.session_state:
        st.session_state.global_annotations = []

    st.sidebar.header("Input Parameters")

    csv_file = st.sidebar.file_uploader("Load Existing Annotations (Optional)", type=["csv"])
    if csv_file is not None and "csv_loaded" not in st.session_state:
        try:
            existing_df = pd.read_csv(csv_file)
            st.session_state.global_annotations.extend(existing_df.to_dict(orient="records"))
            st.session_state.csv_loaded = True  # Marcamos que ya se cargó el CSV
            st.sidebar.success("CSV loaded and annotations appended.")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")

    dataset_folder = st.sidebar.text_input("Dataset Folder", "path/to/your/flowers_dataset")

    query_image_file = st.sidebar.file_uploader("Upload Query Image", type=["jpg", "jpeg", "png"])
    query_image_path = None
    if query_image_file is not None:
        query_image = Image.open(query_image_file)
        query_image_path = f"temp_query_image.{query_image_file.name.split('.')[-1]}"
        query_image.save(query_image_path)

    reference_flower_name = st.sidebar.text_input("Flower Name (Reference)", "")

    retrieve_all = st.sidebar.checkbox("Retrieve All Similar Images", value=False)
    if not retrieve_all:
        top_k = st.sidebar.number_input("Number of Images to Retrieve (Top K)", min_value=1, value=10, step=1)
    else:
        top_k = None

    if st.sidebar.button("Execute Retrieval"):
        if dataset_folder and query_image_path and reference_flower_name:
            try:
                extractor = FeatureExtractor()
                cbir_system = CBIRSystem(dataset_folder, extractor)
                results = cbir_system.retrieve_similar_images(query_image_path, top_k=top_k)
                st.session_state.current_results = results
                st.session_state.current_index = 0
                st.session_state.reference_name = reference_flower_name
                st.success("Retrieval completed! Begin annotation.")
            except Exception as e:
                st.error(f"Error during retrieval: {e}")
        else:
            st.error("Please enter the dataset folder, upload a query image, and specify the flower name.")

    if "current_results" in st.session_state and "reference_name" in st.session_state:
        current_index = st.session_state.current_index
        results = st.session_state.current_results
        reference_name = st.session_state.reference_name

        if current_index < len(results):
            image_path, features, score = results[current_index]
            st.subheader(f"Image {current_index + 1}/{len(results)}")
            st.write(f"Similarity Score: {score:.2f}")
            try:
                pil_image = Image.open(image_path)
                st.image(pil_image, width=400)
            except Exception as e:
                st.error(f"Error loading image {image_path}: {e}")

            # Ask: "Is this image of [flower name]?"
            st.write(f"Is this image of **{reference_name}**?")
            col1, col2 = st.columns(2)
            if col1.button("Yes"):
                st.session_state.global_annotations.append({"image": image_path, "features_vector": features ,"flower_name": reference_name})
                st.session_state.current_index += 1
                st.rerun()
                cbir_system.finetune(query_image_path, [image_path], [])
                st.write("Finetuned")

            if col2.button("No"):
                st.session_state.current_index += 1
                st.rerun()
                cbir_system.finetune(query_image_path, [], [image_path])
                st.write("Finetuned")
        else:
            st.success("Annotation session completed.")
            annotations_df = pd.DataFrame(st.session_state.global_annotations)
            st.write("Accumulated Annotations:")
            st.dataframe(annotations_df)

            csv_data = annotations_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_data, file_name="annotations.csv", mime="text/csv")

            if st.button("New Query"):
                for key in ["current_results", "current_index", "reference_name"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            
            # Remove the query image
            if query_image_path:
                import os
                os.remove(query_image_path)

if __name__ == "__main__":
    main()