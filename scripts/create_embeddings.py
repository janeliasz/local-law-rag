import os
import yaml
import pickle
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


DATASET_DIR = "outputs/dataset"
OUTPUT_PATH = "outputs/embeddings.pkl"

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
    DEVICE = params["device"]
    params = params["create_embeddings"]


def main():
    initial_files = set(os.listdir(DATASET_DIR))
    dataset = load_from_disk(DATASET_DIR)

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}

    model = SentenceTransformer(params["biencoder"])
    model.to(DEVICE)

    filtered_dataset = dataset.filter(lambda row: row["file_name"] not in embeddings)

    if len(filtered_dataset) == 0:
        print("No files to process")
        return

    new_embeddings = (
        model.encode(filtered_dataset["text"], convert_to_tensor=True)
        .detach()
        .cpu()
        .numpy()
    )
    print("Created embeddings array of shape: ", new_embeddings.shape)

    for file_name, embedding in zip(filtered_dataset["file_name"], new_embeddings):
        embeddings[file_name] = embedding

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    # for some reason load_from_disk() creates a cache file in outputs/dataset and dvc gets lost
    final_files = set(os.listdir(DATASET_DIR))
    new_files = final_files - initial_files
    for file in new_files:
        file_path = os.path.join(DATASET_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    main()
