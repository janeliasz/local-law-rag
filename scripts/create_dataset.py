import os
import pdfplumber
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm


INPUT_DIR = "outputs/scraped"
OUTPUT_DIR = "outputs/dataset"


def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return text


def main():
    if os.path.exists(OUTPUT_DIR):
        dataset = Dataset.load_from_disk(OUTPUT_DIR)
        existing_files = set(dataset["file_name"])
    else:
        dataset = None
        existing_files = set()

    new_data = []
    for pdf_file in tqdm(os.listdir(INPUT_DIR)):
        if pdf_file.endswith(".pdf") and pdf_file not in existing_files:
            pdf_path = os.path.join(INPUT_DIR, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            new_data.append({"file_name": pdf_file, "text": text})

    if new_data:
        new_dataset = Dataset.from_list(new_data)

        dataset = (
            concatenate_datasets([dataset, new_dataset]) if dataset else new_dataset
        )

        dataset.save_to_disk(OUTPUT_DIR)


if __name__ == "__main__":
    main()
