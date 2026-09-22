import json
import gc
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

INPUT_FILE = Path(
    "data/chunks_semantic.jsonl"
)

OUTPUT_FILE = Path(
    "data/embeddings/chunks_semantic.jsonl"
)

# Start conservatively for RTX 4060 8GB
BATCH_SIZE = 16

# Maximum number of tokens sent to Qwen.
# This protects against pathological PDF chunks.
MAX_SEQ_LENGTH = 1024

DEVICE = "cuda"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("Loading embedding model")
print("=" * 70)

model = SentenceTransformer(
    MODEL_NAME,
    device=DEVICE
)

# Important:
# Prevent very long PDF-extracted chunks from consuming
# enormous amounts of GPU memory.
model.max_seq_length = MAX_SEQ_LENGTH

print(f"Model       : {MODEL_NAME}")
print(f"Device      : {model.device}")
print(f"Max tokens  : {model.max_seq_length}")
print(f"Batch size  : {BATCH_SIZE}")


# ============================================================
# EMBED ONE BATCH
# ============================================================

def embed_batch(texts, batch_size):

    embeddings = model.encode(
        texts,

        batch_size=batch_size,

        show_progress_bar=False,

        normalize_embeddings=True,

        convert_to_numpy=True,

        # Don't return tensors to GPU
        convert_to_tensor=False
    )

    return embeddings


# ============================================================
# STREAMING EMBEDDING
# ============================================================

def process_file():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Count chunks first
    # --------------------------------------------------------

    print("\nCounting chunks...")

    total_chunks = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():
                total_chunks += 1

    print(
        f"Total semantic chunks: "
        f"{total_chunks:,}"
    )

    # --------------------------------------------------------
    # Streaming processing
    # --------------------------------------------------------

    processed = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as infile, open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as outfile:

        batch_chunks = []

        for line_number, line in enumerate(
            infile,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                chunk = json.loads(line)

            except json.JSONDecodeError as e:

                print(
                    f"\nSkipping invalid JSON "
                    f"at line {line_number}: {e}"
                )

                continue

            batch_chunks.append(chunk)

            # ------------------------------------------------
            # Process batch
            # ------------------------------------------------

            if len(batch_chunks) >= BATCH_SIZE:

                processed = process_batch(
                    batch_chunks,
                    outfile,
                    processed,
                    total_chunks
                )

                batch_chunks = []

        # ----------------------------------------------------
        # Process final partial batch
        # ----------------------------------------------------

        if batch_chunks:

            processed = process_batch(
                batch_chunks,
                outfile,
                processed,
                total_chunks
            )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Processed: {processed:,} / "
        f"{total_chunks:,}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )


# ============================================================
# PROCESS BATCH
# ============================================================

def process_batch(
    chunks,
    outfile,
    processed,
    total
):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    current_batch_size = len(texts)

    # --------------------------------------------------------
    # Try normal batch
    # --------------------------------------------------------

    while True:

        try:

            embeddings = embed_batch(
                texts,
                current_batch_size
            )

            break

        except torch.cuda.OutOfMemoryError:

            print(
                "\nCUDA OOM!"
            )

            print(
                f"Reducing batch size: "
                f"{current_batch_size} -> "
                f"{max(1, current_batch_size // 2)}"
            )

            # Clear GPU memory
            gc.collect()

            torch.cuda.empty_cache()

            current_batch_size = max(
                1,
                current_batch_size // 2
            )

            if current_batch_size == 1:

                # Try one at a time
                embeddings = []

                for chunk in chunks:

                    try:

                        embedding = embed_batch(
                            [chunk["text"]],
                            1
                        )[0]

                        embeddings.append(
                            embedding
                        )

                    except torch.cuda.OutOfMemoryError:

                        print(
                            "\nWARNING:"
                        )

                        print(
                            f"Could not embed "
                            f"chunk {chunk['id']}"
                        )

                        # Empty embedding so the pipeline
                        # doesn't silently crash
                        embeddings.append([])

                        gc.collect()
                        torch.cuda.empty_cache()

                break

            # Retry with smaller batch

    # --------------------------------------------------------
    # Write embeddings immediately
    # --------------------------------------------------------

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        record = {

            "id": chunk["id"],

            "text": chunk["text"],

            "metadata": chunk["metadata"],

            "embedding": embedding.tolist()
            if hasattr(embedding, "tolist")
            else embedding
        }

        outfile.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

    # Flush regularly
    outfile.flush()

    processed += len(chunks)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    percentage = (
        processed / total
    ) * 100

    print(
        f"\rEmbedding: "
        f"{processed:,}/{total:,} "
        f"({percentage:.2f}%)",
        end="",
        flush=True
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del embeddings

    gc.collect()

    return processed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_file()