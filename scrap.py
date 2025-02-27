import os
import argparse
from tqdm import tqdm
import requests


def get_documents_ids():
    url = "https://prawomiejscowe.pl/api/documents"

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
    }

    payload = {"InstitutionId": 548, "AdditionalId": 2024, "SearchForType": 14}

    # payload = {
    #     "PageNumber": 0,
    #     "PageSize": 7,
    #     "SearchText": "",
    #     "SearchTextInPdf": False,
    #     "HideAmendingActs": False,
    #     "Asc": 1,
    #     "ColumnId": -1,
    #     "InstitutionId": 548,
    #     "AdditionalId": 0,
    #     "SearchForType" :22
    # }

    try:
        response = requests.post(url, json=payload, headers=headers)
    except requests.exceptions.RequestException as e:
        print(e)

    data = response.json()

    ids = [doc["Id"] for doc in data["Documents"]]

    return ids


def get_document_details(document_id):
    url = f"https://prawomiejscowe.pl/api/documents/548/details/{document_id}"

    try:
        response = requests.get(url)
    except requests.exceptions.RequestException as e:
        print(e)

    return response.json()


def download_document_files(files, output_dir):
    for file in files:
        if file["Name"].endswith(".pdf"):
            download_url = f"https://prawomiejscowe.pl/api/file/548/{file["Id"]}"
            file_path = os.path.join(output_dir, f"{file["Id"]}.pdf")

            if os.path.exists(file_path):
                print(f"File {file_path} already exists")
                continue

            try:
                response = requests.get(download_url, stream=True)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {file_path}")

            except requests.exceptions.RequestException as e:
                print(e)

        if len(file["Children"]) > 0:
            download_document_files(file["Children"], output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default="outputs/scraped"
    )

    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir

    document_ids = get_documents_ids()

    for document_id in tqdm(document_ids):
        details = get_document_details(document_id)

        download_document_files(details["Files"], OUTPUT_DIR)


if __name__ == "__main__":
    main()
