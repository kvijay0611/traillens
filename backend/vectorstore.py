"""
Local, file-persisted vector store using Chroma + sentence-transformers.
No external vector DB service required, so it deploys as a single web
service on Render with zero extra infra. Swappable for pgvector/Pinecone --
see README "Scaling this up" section.
"""
import chromadb
from chromadb.utils import embedding_functions
from backend.config import CHROMA_PATH

_client = None
_collection = None
_embedder = None

# Cosine distance cutoff: chunks farther than this from the query are treated
# as irrelevant and dropped, instead of being passed to the LLM as "context."
# Without this, a semantically weak match from a totally different trial can
# get pulled in just because it's the "closest of what's available" -- which
# is how unrelated NCT IDs from earlier questions ended up cited on a NASH
# query. 0.9 is a reasonably strict cutoff for all-MiniLM-L6-v2 embeddings.
RELEVANCE_DISTANCE_THRESHOLD = 0.9


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return _embedder


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name="trial_chunks", embedding_function=_get_embedder()
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
    """
    Returns up to k semantically relevant chunks, filtered by a distance
    threshold so stale/unrelated chunks from previous unrelated questions
    don't leak into the current answer.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []
    # Over-fetch a bit before filtering, since some results will get dropped.
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