from pathlib import Path
import json

from langchain_community.document_loaders import PyPDFLoader


BOOKS_DIR = Path("Books")
OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "documents.json"


def load_pdfs():
    documents = []
    failed_files = []

    pdf_files = sorted(BOOKS_DIR.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files\n")

    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] Loading: {pdf_path.name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

            # Enrich basic metadata
            for doc in docs:
                doc.metadata["source"] = pdf_path.name
                doc.metadata["book_name"] = pdf_path.stem

                # PyPDFLoader usually gives page as 0-indexed
                if "page" in doc.metadata:
                    doc.metadata["page_no"] = doc.metadata["page"] + 1

            documents.extend(docs)

            print(f"    ✓ {len(docs)} pages")

        except Exception as e:
            failed_files.append({
                "file": str(pdf_path),
                "error": str(e),
            })

            print(f"    ✗ FAILED: {e}")

    return documents, failed_files


def save_documents(documents, failed_files):
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = {
        "documents": [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ],
        "failed_files": failed_files,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n==============================")
    print(f"Loaded pages : {len(documents)}")
    print(f"Failed PDFs  : {len(failed_files)}")
    print(f"Saved to     : {OUTPUT_FILE}")
    print("==============================")


if __name__ == "__main__":
    documents, failed_files = load_pdfs()
    save_documents(documents, failed_files)