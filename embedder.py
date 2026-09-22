import json
from pathlib import Path

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

INPUT_FILES = [
    Path("data/chunks_recursive.jsonl"),
    Path("data/chunks_semantic.jsonl"),
]

OUTPUT_DIR = Path("data/embeddings")

BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)

print(f"Model: {MODEL_NAME}")
print(f"Device: {model.device}")


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(file_path):

    chunks = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            chunks.append(
                json.loads(line)
            )

    return chunks


# ============================================================
# EMBEDDING
# ============================================================

def embed_chunks(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Generating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(
        texts,

        batch_size=BATCH_SIZE,

        show_progress_bar=True,

        normalize_embeddings=True,

        convert_to_numpy=True
    )

    return embeddings


# ============================================================
# SAVE
# ============================================================

def save_embeddings(
    chunks,
    embeddings,
    output_file
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            record = {
                "id": chunk["id"],

                "text": chunk["text"],

                "metadata": chunk["metadata"],

                "embedding": embedding.tolist()
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for input_file in INPUT_FILES:

        print("\n" + "=" * 70)
        print(f"Processing: {input_file}")
        print("=" * 70)

        # ----------------------------------------------------
        # Load chunks
        # ----------------------------------------------------

        chunks = load_jsonl(
            input_file
        )

        print(
            f"Loaded {len(chunks)} chunks"
        )

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        embeddings = embed_chunks(
            chunks
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_file = (
            OUTPUT_DIR / input_file.name
        )

        save_embeddings(
            chunks,
            embeddings,
            output_file
        )

        print(
            f"Saved: {output_file}"
        )

        print(
            f"Embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    print("\nAll embeddings generated.")