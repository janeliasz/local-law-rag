import pickle
import numpy as np
from datasets import Dataset


def get_faiss_dataset(dataset_dir: str, embeddings_path: str) -> Dataset:
    dataset = Dataset.load_from_disk(dataset_dir)

    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)

    dataset = dataset.map(
        lambda batch: {
            "embedding": [
                embeddings[file_name].tolist() for file_name in batch["file_name"]
            ]
        },
        batched=True,
    )

    dataset.add_faiss_index(column="embedding")

    return dataset


def retrieve(model, faiss_dataset, query, top_k=20):
    query_embedding = model.encode(query)

    _, samples = faiss_dataset.get_nearest_examples(
        "embedding", query_embedding, k=top_k
    )

    return samples


def rerank(reranker, samples, query, top_k=5):
    reranker_input = [[query, text] for text in samples["text"]]

    reranker_scores = reranker.predict(reranker_input)

    ranking = np.argsort(reranker_scores)[::-1]

    top_k_indices = ranking[:top_k]

    ranked_samples = {
        "file_name": [samples["file_name"][i] for i in top_k_indices],
        "text": [samples["text"][i] for i in top_k_indices],
    }

    return ranked_samples
