import os
import argparse
import pickle
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory with dataset",
        default="outputs/dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for embeddings",
        default="outputs/embeddings.pkl",
    )
    parser.add_argument("--device", type=str, default="mps")

    args = parser.parse_args()
    DATASET_DIR = args.dataset_dir
    OUTPUT_PATH = args.output_path
    DEVICE = args.device

    dataset = load_from_disk(DATASET_DIR)

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = {}

    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
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


if __name__ == "__main__":
    main()
