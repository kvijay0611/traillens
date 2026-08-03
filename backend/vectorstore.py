"""
Lightweight retrieval layer -- NOT a real vector store.

This started as a Chroma + sentence-transformers RAG pipeline, but Render's
free-tier instances (512MB RAM) kept getting OOM-killed: PyTorch alone used
400MB+, and even after switching to Chroma's built-in ONNX embedder, the
combination of onnxruntime + chromadb + FastAPI + LangGraph + the OpenAI SDK
still exceeded the limit.

For a free-tier deployment, this trades embedding-based semantic search for
a dependency-free TF-IDF-style keyword overlap scorer implemented in plain
Python -- no numpy, no ML runtime, negligible memory footprint. It's a real
limitation worth naming explicitly in an interview: this retrieves based on
term overlap and weighting, not learned semantic similarity, so it will miss
paraphrases a real embedding model would catch (e.g. "tumor shrinkage" vs
"reduction in lesion size").

For a production deployment with adequate memory (or a managed embeddings
API instead of a local model), swap this module for pgvector + a hosted
embeddings endpoint -- the semantic_search()/index_trial() interface below
is designed to be a drop-in replacement target. See README "Scaling this up".
"""
import math
import re
from collections import Counter
from backend.db import get_session, TrialChunk

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "for", "to", "with", "and", "or", "but", "what",
    "which", "who", "whom", "this", "that", "these", "those", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "any", "trials", "trial", "criteria", "eligibility",
}


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def index_trial(trial: dict):
    """Chunk a trial's free-text fields and store them for keyword retrieval."""
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

    session = get_session()
    try:
        # Replace any existing chunks for this trial (keeps re-ingestion idempotent).
        session.query(TrialChunk).filter(TrialChunk.nct_id == nct_id).delete()
        for section, text in chunks:
            session.add(TrialChunk(
                nct_id=nct_id,
                section=section,
                text=text,
                title=trial.get("title", ""),
                condition=trial.get("condition", ""),
                phase=trial.get("phase", ""),
                status=trial.get("status", ""),
            ))
        session.commit()
    finally:
        session.close()


def semantic_search(query: str, k: int = 5) -> list[dict]:
    """
    Returns up to k chunks ranked by TF-IDF-weighted term overlap with the
    query. Chunks with zero overlapping terms are excluded entirely (this is
    the equivalent of the distance-threshold cutoff a real vector search
    would apply, to keep unrelated chunks from earlier queries out of the
    current answer).
    """
    query_terms = set(_tokenize(query))
    if not query_terms:
        return []

    session = get_session()
    try:
        all_chunks = session.query(TrialChunk).all()
        if not all_chunks:
            return []

        # Document frequency across the local corpus, for IDF weighting.
        doc_term_sets = [set(_tokenize(c.text)) for c in all_chunks]
        n_docs = len(doc_term_sets)
        df = Counter()
        for terms in doc_term_sets:
            for t in terms & query_terms:
                df[t] += 1

        scored = []
        for chunk, terms in zip(all_chunks, doc_term_sets):
            overlap = terms & query_terms
            if not overlap:
                continue
            term_counts = Counter(_tokenize(chunk.text))
            score = 0.0
            for t in overlap:
                tf = term_counts[t]
                idf = math.log((n_docs + 1) / (df[t] + 1)) + 1
                score += tf * idf
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        return [
            {
                "text": chunk.text,
                "metadata": {
                    "nct_id": chunk.nct_id,
                    "title": chunk.title,
                    "condition": chunk.condition,
                    "phase": chunk.phase,
                    "status": chunk.status,
                    "section": chunk.section,
                },
                "score": score,
            }
            for score, chunk in top
        ]
    finally:
        session.close()
