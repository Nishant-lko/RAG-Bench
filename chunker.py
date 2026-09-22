import json
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIG
# ============================================================

DOCUMENTS_FILE = Path("data/documents.json")

RECURSIVE_OUTPUT = Path("data/chunks_recursive.jsonl")
SEMANTIC_OUTPUT = Path("data/chunks_semantic.jsonl")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Recursive chunking
RECURSIVE_CHUNK_SIZE = 1000
RECURSIVE_CHUNK_OVERLAP = 150

# Semantic chunking
SIMILARITY_THRESHOLD = 0.70
SEMANTIC_MAX_CHARS = 2000


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    if not DOCUMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {DOCUMENTS_FILE}"
        )

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
        documents = json.load(f)

    return documents


# ============================================================
# SENTENCE SPLITTER
# ============================================================

def split_sentences(text):
    """
    Simple sentence splitter.

    Splits after:
        .
        !
        ?
    
    while preserving the sentence text.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip()
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ============================================================
# RECURSIVE CHUNKING
# ============================================================

def recursive_chunking(
    documents,
    chunk_size=1000,
    chunk_overlap=150
):
    """
    Chunk documents using LangChain's
    RecursiveCharacterTextSplitter.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,

        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            ""
        ],

        length_function=len
    )

    chunks = []

    chunk_id = 0

    for document_index, document in enumerate(documents):

        text = document["text"]
        metadata = document.get("metadata", {})

        if not text.strip():
            continue

        split_texts = splitter.split_text(text)

        for chunk_text in split_texts:

            chunk = {
                "id": f"recursive_{chunk_id}",

                "text": chunk_text,

                "metadata": {
                    **metadata,

                    "chunking_method": "recursive",

                    "source_document": document_index,

                    "chunk_size": len(chunk_text)
                }
            }

            chunks.append(chunk)

            chunk_id += 1

    return chunks


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def semantic_chunking(
    documents,
    model,
    similarity_threshold=0.70,
    max_chars=2000
):
    """
    Semantic chunking based on cosine similarity
    between consecutive sentences.

    If similarity between two consecutive sentences
    drops below the threshold, a new chunk is started.
    """

    chunks = []

    chunk_id = 0

    for document_index, document in enumerate(documents):

        text = document["text"]
        metadata = document.get("metadata", {})

        if not text.strip():
            continue

        # ----------------------------------------------------
        # 1. Split document into sentences
        # ----------------------------------------------------

        sentences = split_sentences(text)

        if not sentences:
            continue

        # ----------------------------------------------------
        # 2. Generate temporary embeddings
        # ----------------------------------------------------

        embeddings = model.encode(
            sentences,

            normalize_embeddings=True,

            show_progress_bar=False,

            convert_to_tensor=True
        )

        # ----------------------------------------------------
        # 3. Build semantic chunks
        # ----------------------------------------------------

        current_chunk = [sentences[0]]

        current_chars = len(sentences[0])

        for i in range(1, len(sentences)):

            previous_embedding = embeddings[i - 1]
            current_embedding = embeddings[i]

            # Because embeddings are normalized,
            # dot product = cosine similarity.
            similarity = (
                previous_embedding @ current_embedding
            ).item()

            sentence = sentences[i]

            sentence_length = len(sentence)

            # ------------------------------------------------
            # Decide whether to continue current chunk
            # ------------------------------------------------

            should_split = (

                similarity < similarity_threshold

                or

                current_chars + sentence_length > max_chars
            )

            if should_split:

                # --------------------------------------------
                # Save current chunk
                # --------------------------------------------

                chunk_text = " ".join(current_chunk)

                chunks.append({

                    "id": f"semantic_{chunk_id}",

                    "text": chunk_text,

                    "metadata": {
                        **metadata,

                        "chunking_method": "semantic",

                        "source_document": document_index,

                        "chunk_size": len(chunk_text)
                    }

                })

                chunk_id += 1

                # --------------------------------------------
                # Start new chunk
                # --------------------------------------------

                current_chunk = [sentence]

                current_chars = sentence_length

            else:

                current_chunk.append(sentence)

                current_chars += sentence_length + 1

        # ----------------------------------------------------
        # 4. Save final chunk
        # ----------------------------------------------------

        if current_chunk:

            chunk_text = " ".join(current_chunk)

            chunks.append({

                "id": f"semantic_{chunk_id}",

                "text": chunk_text,

                "metadata": {
                    **metadata,

                    "chunking_method": "semantic",

                    "source_document": document_index,

                    "chunk_size": len(chunk_text)
                }

            })

            chunk_id += 1

    return chunks


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(chunks, output_file):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# PRINT SAMPLE CHUNKS
# ============================================================

def inspect_chunks(chunks, title, number=3):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for i, chunk in enumerate(chunks[:number]):

        print(f"\nCHUNK {i}")

        print("ID:")
        print(chunk["id"])

        print("\nMetadata:")
        print(chunk["metadata"])

        print("\nText:")
        print(chunk["text"][:1000])

        print("\n" + "-" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 1. Load documents
    # ========================================================

    print("Loading documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents/pages"
    )


    # ========================================================
    # 2. RECURSIVE CHUNKING
    # ========================================================

    print("\n")
    print("=" * 80)
    print("RUNNING RECURSIVE CHUNKING")
    print("=" * 80)

    recursive_chunks = recursive_chunking(
        documents,

        chunk_size=RECURSIVE_CHUNK_SIZE,

        chunk_overlap=RECURSIVE_CHUNK_OVERLAP
    )

    print(
        f"Created {len(recursive_chunks)} recursive chunks"
    )


    # ========================================================
    # 3. Save recursive chunks
    # ========================================================

    save_jsonl(
        recursive_chunks,
        RECURSIVE_OUTPUT
    )

    print(
        f"Saved recursive chunks to: "
        f"{RECURSIVE_OUTPUT}"
    )


    # ========================================================
    # 4. Load embedding model ONCE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("LOADING EMBEDDING MODEL")
    print("=" * 80)

    model = SentenceTransformer(
        EMBEDDING_MODEL,

        device="cuda"
    )

    print(
        f"Model device: {model.device}"
    )


    # ========================================================
    # 5. SEMANTIC CHUNKING
    # ========================================================

    print("\n")
    print("=" * 80)
    print("RUNNING SEMANTIC CHUNKING")
    print("=" * 80)

    semantic_chunks = semantic_chunking(

        documents,

        model,

        similarity_threshold=SIMILARITY_THRESHOLD,

        max_chars=SEMANTIC_MAX_CHARS
    )

    print(
        f"Created {len(semantic_chunks)} semantic chunks"
    )


    # ========================================================
    # 6. Save semantic chunks
    # ========================================================

    save_jsonl(
        semantic_chunks,
        SEMANTIC_OUTPUT
    )

    print(
        f"Saved semantic chunks to: "
        f"{SEMANTIC_OUTPUT}"
    )


    # ========================================================
    # 7. Compare basic statistics
    # ========================================================

    print("\n")
    print("=" * 80)
    print("CHUNKING COMPARISON")
    print("=" * 80)

    print(
        f"Documents              : {len(documents)}"
    )

    print(
        f"Recursive chunks       : {len(recursive_chunks)}"
    )

    print(
        f"Semantic chunks        : {len(semantic_chunks)}"
    )


    # ========================================================
    # 8. Inspect examples
    # ========================================================

    inspect_chunks(
        recursive_chunks,

        "RECURSIVE CHUNKING — SAMPLE"
    )

    inspect_chunks(
        semantic_chunks,

        "SEMANTIC CHUNKING — SAMPLE"
    )


    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)