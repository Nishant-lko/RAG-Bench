from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


client = QdrantClient(path="data/qdrant")

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cuda"
)

model.max_seq_length = 1024


def retrieve(query, collection, k=5):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    results = client.query_points(
        collection_name=collection,
        query=query_embedding.tolist(),
        limit=k,
        with_payload=True
    ).points

    output = []

    for result in results:

        payload = result.payload

        output.append({
            "score": result.score,
            "text": payload["text"],
            "metadata": payload["metadata"]
        })

    return output