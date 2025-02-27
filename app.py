import streamlit as st
import os
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder

from rag import get_faiss_dataset, retrieve, rerank


DEVICE = "mps"
RETRIEVER_TOP_K = 50
RERANKER_TOP_K = 10

CLARIN_KEY = os.environ["CLARIN_KEY"]
CLARIN_URL = "https://services.clarin-pl.eu/api/v1/oapi/chat/completions"

PROMPT_TEMPLATE_RAG = """You are a helpful assistant. Answer the following QUERY utilizing provided CONTEXT.

QUERY: {query}

CONTEXT: {context}"""


def send_request(message):
    headers = {
        "Authorization": f"Bearer {CLARIN_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "bielik",
        "messages": [
            {"role": "user", "content": message},
        ],
    }

    try:
        response = requests.post(CLARIN_URL, json=payload, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Request error: {e}")
        return None

    return response.json()


@st.cache_resource
def load_dataset():
    return get_faiss_dataset("outputs/dataset", "outputs/embeddings.pkl")


@st.cache_resource
def load_models():
    biencoder = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    biencoder.to(DEVICE)

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker.model.to(DEVICE)
    return biencoder, reranker


st.markdown("<title>Local Law RAG - Document Search</title>", unsafe_allow_html=True)
st.title("Local Law RAG")

query = st.text_input("Enter your query: ")

if query:
    with st.spinner("Searching for relevant documents..."):
        dataset = load_dataset()
        biencoder, reranker = load_models()

        samples = retrieve(
            model=biencoder, faiss_dataset=dataset, query=query, top_k=RETRIEVER_TOP_K
        )
        ranked_samples = rerank(
            reranker=reranker, samples=samples, query=query, top_k=RERANKER_TOP_K
        )

    context = "\n".join(
        [
            f"Document {i+1}: {document}"
            for i, document in enumerate(ranked_samples["text"])
        ]
    )
    prompt = PROMPT_TEMPLATE_RAG.format(query=query, context=context)

    with st.spinner("Generating model's response..."):
        response = send_request(prompt)
        answer = response["choices"][0]["message"]["content"]
        answer = answer.strip()

    if answer:
        st.subheader("Model's Answer:")
        st.write(answer)
    else:
        st.error("Failed to get an answer from the model.")

    st.header(f"Top {RERANKER_TOP_K} ranked documents:")
    st.write("")

    for i, file_name in enumerate(ranked_samples["file_name"]):
        pdf_path = os.path.join("outputs/scraped", file_name)

        text_preview = ranked_samples["text"][i][:256]

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                pdf_data = pdf_file.read()

            st.download_button(
                label=f"**{i + 1}.** {file_name} (Download PDF)",
                data=pdf_data,
                file_name=file_name,
                mime="application/pdf",
            )

            st.write(f"**Text preview:** {text_preview}...")

        else:
            st.write(f"**{i + 1}.** {file_name} (PDF not found)")

        st.write("---")
