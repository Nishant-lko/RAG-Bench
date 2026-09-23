from fastapi import FastAPI
from pydantic import BaseModel

from app.retrieval import retrieve


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "ChunkBench API running"}


@app.post("/retrieve")
def get_context(request: QueryRequest):

    recursive = retrieve(
        request.query,
        "recursive_chunks",
        k=5
    )

    semantic = retrieve(
        request.query,
        "semantic_chunks",
        k=5
    )

    return {
        "recursive": recursive,
        "semantic": semantic
    }