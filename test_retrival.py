from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


client = QdrantClient(
    path="data/qdrant"
)

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cuda"
)

model.max_seq_length = 1024


query = input("\nEnter your question: ")

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)


results = client.query_points(
    collection_name="recursive_chunks",

    query=query_embedding.tolist(),

    limit=5,

    with_payload=True
).points


print("\n" + "=" * 70)
print("RECURSIVE CHUNKING — TOP 5 RESULTS")
print("=" * 70)


for i, result in enumerate(results, start=1):

    payload = result.payload

    print(f"\n[{i}] Score: {result.score:.4f}")

    print(f"ID: {payload['id']}")

    print(f"Metadata: {payload['metadata']}")

    print("\nContext:")
    print(payload["text"][:1000])

    print("-" * 70)