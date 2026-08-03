"""
Local, file-persisted vector store using Chroma with its built-in ONNX
MiniLM embedder (NOT sentence-transformers/PyTorch, which is far too heavy
for a 512MB free-tier instance -- PyTorch alone can eat 400MB+ of RAM before
processing a single request, and Render's free web services will OOM-kill
the process). Chroma's default embedding function does the same job with a
much smaller footprint.
"""
import chromadb
from chromadb.utils import embedding_functions
from backend.config import CHROMA_PATH

_client = None
_collection = None

# Lightweight ONNX-based embedder built into chromadb -- no torch required.
_embedder = embedding_functions.DefaultEmbeddingFunction()

RELEVANCE_DISTANCE_THRESHOLD = 0.9


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name="trial_chunks", embedding_function=_embedder
        )
    return _collection


def index_trial(trial: dict):
    """Chunk a trial's free-text fields and upsert into the vector store."""
    collection = get_collection()
    nct_id = trial["nct_id"]
    chunks = []

    if trial.get("brief_summary"):
        chunks.append(("summary", trial["brief_summary"]))
    if trial.get("eligibility_criteria"):
        parts = [p.strip() for p in trial["eligibility_criteria"].split("\n\n") if p.strip()]
        for i, p in enumerate(parts):
            chunks.append((f"eligibility_{i}", p))

    if not chunks:
        return

    ids = [f"{nct_id}::{tag}" for tag, _ in chunks]
    docs = [text for _, text in chunks]
    metadatas = [
        {
            "nct_id": nct_id,
            "title": trial.get("title", ""),
            "condition": trial.get("condition", ""),
            "phase": trial.get("phase", ""),
            "status": trial.get("status", ""),
            "section": tag,
        }
        for tag, _ in chunks
    ]
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)


def semantic_search(query: str, k: int = 5) -> list[dict]:
    collection = get_collection()
    if collection.count() == 0:
        return []
    n = min(k * 3, collection.count())
    results = collection.query(query_texts=[query], n_results=n)

    out = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        if dist > RELEVANCE_DISTANCE_THRESHOLD:
            continue
        out.append({"text": doc, "metadata": meta, "distance": dist})
        if len(out) >= k:
            break
    return out