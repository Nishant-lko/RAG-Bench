# ChunkBench

**ChunkBench** is a small experimental framework for studying and comparing document chunking strategies used in Retrieval-Augmented Generation (RAG) systems.

The project currently focuses on comparing:

* Recursive Character Chunking
* Semantic Chunking

The goal is to understand how different chunking strategies affect the structure and quality of the resulting document chunks before they are embedded and stored in a vector database.

## Pipeline

```text
PDF
 ↓
PyPDFLoader
 ↓
Document Extraction
 ↓
┌─────────────────────┐
│      Chunking       │
├─────────────────────┤
│ Recursive Chunking  │
│ Semantic Chunking   │
└──────────┬──────────┘
           ↓
      Compare Chunks
           ↓
     Final Embeddings
           ↓
       Vector DB
           ↓
          RAG
```

## Current Features

* PDF document loading
* Document preprocessing
* Recursive chunking
* Semantic chunking using sentence embeddings
* JSONL output for generated chunks
* Basic comparison of chunking strategies

## Tech Stack

* Python
* PyPDF
* LangChain Text Splitters
* Sentence Transformers
* PyTorch
* CUDA

## Project Structure

```text
ChunkBench/
│
├── data/
│   ├── documents.json
│   ├── chunks_recursive.jsonl
│   └── chunks_semantic.jsonl
│
├── loader.py
├── chunker.py
└── README.md
```

## Running

Install dependencies:

```bash
pip install langchain-text-splitters sentence-transformers
```

Run the chunking pipeline:

```bash
python chunker.py
```

The generated chunks will be saved in:

```text
data/chunks_recursive.jsonl
data/chunks_semantic.jsonl
```

## Future Work

* Add more chunking strategies
* Benchmark chunking methods
* Evaluate retrieval performance
* Add embeddings and vector database
* Build a complete RAG pipeline
* Add visualization and experiment tracking

## Status

**Early development / experimentation**
