import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


INPUT_FILE = "data/embeddings/chunks_recursive.jsonl"
COLLECTION = "recursive_chunks"

BATCH_SIZE = 256


# --------------------------------------------------
# Connect to Docker Qdrant
# --------------------------------------------------

client = QdrantClient(
    url="http://localhost:6333"
)


# --------------------------------------------------
# Determine embedding dimension
# --------------------------------------------------

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    first_record = json.loads(f.readline())

vector_dim = len(first_record["embedding"])

print(f"Embedding dimension: {vector_dim}")


# --------------------------------------------------
# Create collection
# --------------------------------------------------

client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(
        size=vector_dim,
        distance=Distance.COSINE,
    ),
)

print(f"Created collection: {COLLECTION}")


# --------------------------------------------------
# Stream embeddings into Qdrant
# --------------------------------------------------

batch = []

uploaded = 0
point_id = 0


with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        record = json.loads(line)

        point = PointStruct(
            id=point_id,
            vector=record["embedding"],
            payload={
                "original_id": record["id"],
                "text": record["text"],
                "metadata": record["metadata"],
            },
        )

        batch.append(point)

        point_id += 1


        # Upload every 256 points
        if len(batch) >= BATCH_SIZE:

            client.upsert(
                collection_name=COLLECTION,
                points=batch,
                wait=True,
            )

            uploaded += len(batch)

            print(
                f"Uploaded: {uploaded}",
                flush=True,
            )

            batch.clear()


# --------------------------------------------------
# Upload remaining points
# --------------------------------------------------

if batch:

    client.upsert(
        collection_name=COLLECTION,
        points=batch,
        wait=True,
    )

    uploaded += len(batch)


# --------------------------------------------------
# Verify
# --------------------------------------------------

print()
print(f"Finished. Uploaded: {uploaded}")

info = client.get_collection(COLLECTION)

print(f"Collection points: {info.points_count}")
