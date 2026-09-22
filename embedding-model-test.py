from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

print("Model loaded")
print("Device:", model.device)

embedding = model.encode(
    "Transformers use self-attention."
)

print("Embedding dimension:", len(embedding))