"""
Agent graph:

    START -> extract_entities -> structured_lookup -> semantic_lookup
          -> adverse_event_lookup -> synthesize -> END

extract_entities uses the LLM to turn a free-text question into structured
filters (condition, drug, phase, wants_adverse_events). Each subsequent node
only does work if the relevant entity was found, so a purely semantic
question skips the adverse-event API call, etc. synthesize is the only node
that produces user-facing text, and it must cite trial NCT IDs it used.
"""
import json
import re
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from backend import llm_client
from backend.db import get_session, Trial
from backend import vectorstore, ingest


class AgentState(TypedDict, total=False):
    query: str
    condition: Optional[str]
    drug: Optional[str]
    phase: Optional[str]
    status: Optional[str]
    wants_adverse_events: bool
    trials: list
    semantic_chunks: list
    adverse_events: list
    answer: str
    citations: list
    route: str
    usage: dict


EXTRACT_SYSTEM = """You extract structured search filters from a clinical-trial question.
Respond ONLY with valid JSON, no markdown fences, no preamble. Schema:
{
  "condition": string or null,   // disease/condition mentioned, e.g. "NASH", "type 2 diabetes"
  "drug": string or null,        // drug or drug class mentioned, e.g. "GLP-1", "semaglutide"
  "phase": string or null,       // trial phase if mentioned, ALWAYS as a single digit: "1", "2", "3", or "4" -- never roman numerals, never the word "Phase"
  "status": string or null,      // trial status if mentioned, e.g. "RECRUITING", "COMPLETED"
  "wants_adverse_events": boolean // true if the user is asking about side effects, safety, adverse events
}
If the question mentions any disease, condition, syndrome, or abbreviation (even if you don't
recognize it), you MUST put it in "condition" -- never leave it null just because you're
unsure what it stands for."""

# Roman-numeral phases occasionally slip through despite the prompt instruction above;
# normalize both sides of the phase comparison so "Phase III" and "PHASE3" still match.
ROMAN_TO_DIGIT = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}


def _normalize_phase(raw: str) -> str:
    if not raw:
        return ""
    s = raw.lower().replace("phase", "").strip()
    return ROMAN_TO_DIGIT.get(s, s)


def _status_matches(target: str, raw: str) -> bool:
    if not target or not raw:
        return False
    target_norm = target.strip().upper().replace(" ", "_")
    status = raw.strip().upper().replace(" ", "_")
    if target_norm == "RECRUITING":
        if "NOT_RECRUITING" in status or status == "ACTIVE_NOT_RECRUITING":
            return False
        return status == "RECRUITING" or status == "ENROLLING_BY_INVITATION" or status == "ACTIVE_RECRUITING" or "RECRUIT" in status
    return target_norm in status


def _chunk_matches_filters(chunk: dict, phase: Optional[str], status: Optional[str]) -> bool:
    metadata = chunk.get("metadata", {})
    if phase:
        target = _normalize_phase(phase)
        if not target or target not in _normalize_phase(metadata.get("phase", "")):
            return False
    if status:
        if not _status_matches(status, metadata.get("status", "")):
            return False
    return True


def extract_entities(state: AgentState) -> AgentState:
    raw, usage_delta = llm_client.complete(EXTRACT_SYSTEM, state["query"], max_tokens=300)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[4:] if raw.lower().startswith("json") else raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"condition": None, "drug": None, "phase": None, "status": None, "wants_adverse_events": False}

    condition = parsed.get("condition")
    drug = parsed.get("drug")
    status = parsed.get("status")

    # Fallback: if the model failed to extract a condition but the question
    # clearly wants trial data (mentions "trial(s)", "recruiting", "phase",
    # "study/studies"), use the raw query as the search term instead of
    # silently returning nothing. fetch_trials already falls back to a
    # full-text query.term search, so this degrades gracefully.
    if not condition and not drug:
        if re.search(r"\btrial|recruit|phase|stud(y|ies)\b", state["query"], re.IGNORECASE):
            condition = state["query"]

    # Recognize a recruiting request even if the model did not output a status.
    if not status:
        if re.search(r"\b(recruit(ing|ed)?|enroll(ing|ment)?|actively recruiting|currently recruiting)\b", state["query"], re.IGNORECASE):
            status = "RECRUITING"
        elif re.search(r"\b(not yet recruiting)\b", state["query"], re.IGNORECASE):
            status = "NOT_YET_RECRUITING"

    print(f"[extract_entities] condition={condition!r} drug={drug!r} phase={parsed.get('phase')!r} status={status!r} "
          f"wants_ae={parsed.get('wants_adverse_events')!r}")

    usage = state.get("usage", {"input_tokens": 0, "output_tokens": 0})
    usage["input_tokens"] += usage_delta["input_tokens"]
    usage["output_tokens"] += usage_delta["output_tokens"]

    return {
        **state,
        "condition": condition,
        "drug": drug,
        "phase": parsed.get("phase"),
        "status": status,
        "wants_adverse_events": bool(parsed.get("wants_adverse_events")),
        "usage": usage,
    }


def structured_lookup(state: AgentState) -> AgentState:
    condition = state.get("condition")
    trials = []
    if condition:
        session = get_session()
        try:
            existing = (
                session.query(Trial)
                .filter(Trial.condition.ilike(f"%{condition}%"))
                .limit(100)
                .all()
            )
            phase = state.get("phase")
            status_filter = state.get("status")
            if len(existing) < 3 or phase or status_filter:
                try:
                    fresh = ingest.fetch_trials(condition)
                    ingest.persist_trials(fresh)
                    for t in fresh:
                        vectorstore.index_trial(t)
                except Exception as e:
                    print(f"[structured_lookup] live ingest failed for '{condition}': {e}")
                existing = (
                    session.query(Trial)
                    .filter(Trial.condition.ilike(f"%{condition}%"))
                    .limit(100)
                    .all()
                )

            q = existing
            phase = state.get("phase")
            if phase:
                target = _normalize_phase(phase)
                filtered = [t for t in q if target and target in _normalize_phase(t.phase or "")]
                if filtered:
                    q = filtered
                else:
                    print(f"[structured_lookup] phase filter '{phase}' matched 0 of {len(q)} trials, returning zero filtered trials")
                    q = []

            status_filter = state.get("status")
            if status_filter:
                filtered = [t for t in q if _status_matches(status_filter, t.status or "")]
                if filtered:
                    q = filtered
                else:
                    print(f"[structured_lookup] status filter '{status_filter}' matched 0 of {len(q)} trials, returning zero filtered trials")
                    q = []

            trials = [
                {
                    "nct_id": t.nct_id,
                    "title": t.title,
                    "phase": t.phase,
                    "status": t.status,
                    "intervention": t.intervention,
                    "brief_summary": (t.brief_summary or "")[:500],
                    "sponsor": t.sponsor,
                }
                for t in q[:15]
            ]
        finally:
            session.close()
    return {**state, "trials": trials}


def semantic_lookup(state: AgentState) -> AgentState:
    chunks = vectorstore.semantic_search(state["query"], k=10)
    phase = state.get("phase")
    status = state.get("status")
    if phase or status:
        filtered = [c for c in chunks if _chunk_matches_filters(c, phase, status)]
        if filtered:
            chunks = filtered
        else:
            print(f"[semantic_lookup] phase/status filter resulted in 0 chunks (phase={phase!r} status={status!r})")
            chunks = []
    return {**state, "semantic_chunks": chunks[:5]}


def adverse_event_lookup(state: AgentState) -> AgentState:
    events = []
    drug = state.get("drug")
    if state.get("wants_adverse_events") and drug:
        try:
            events = ingest.fetch_adverse_events(drug, limit=8)
        except Exception as e:
            print(f"[adverse_event_lookup] FAERS lookup failed for '{drug}': {e}")
    return {**state, "adverse_events": events}


SYNTH_SYSTEM = """You are TrialLens, a clinical trial research assistant.
Answer the user's question using ONLY the structured trial data, retrieved
document excerpts, and adverse event data provided in the context below.
Rules:
- Always cite NCT IDs in parentheses when referencing a specific trial, e.g. (NCT01234567).
- If the context says no data was found, or is empty, you MUST respond with
  something like: "I did not find any matching trials in the available data."
  or "The available context does not include any Phase 3 trials recruiting for
  this condition." Do NOT answer from your own general knowledge in this case,
  even partially -- a wrong or outdated trial detail is a safety issue.
- If the context contains any structured trial data or retrieved document excerpts,
  do NOT say you couldn't retrieve live data; answer directly from the available
  context and cite the trials provided.
- Never blend "external knowledge" or prior training data into the answer.
  Every factual claim must trace back to the provided context.
- Keep the answer concise and clinically precise. Use short paragraphs or bullet points.
- This is not medical advice; do not make treatment recommendations."""


def synthesize(state: AgentState) -> AgentState:
    context_parts = []
    citations = []

    if state.get("trials"):
        context_parts.append("STRUCTURED TRIAL DATA:\n" + json.dumps(state["trials"], indent=2))
        citations.extend([t["nct_id"] for t in state["trials"]])

    if state.get("semantic_chunks"):
        sem_text = "\n\n".join(
            f"[{c['metadata']['nct_id']} - {c['metadata']['section']}]: {c['text'][:400]}"
            for c in state["semantic_chunks"]
        )
        context_parts.append("RETRIEVED DOCUMENT EXCERPTS:\n" + sem_text)
        citations.extend([c["metadata"]["nct_id"] for c in state["semantic_chunks"]])

    if state.get("adverse_events"):
        context_parts.append("ADVERSE EVENT DATA (FDA FAERS):\n" + json.dumps(state["adverse_events"], indent=2))

    context = "\n\n---\n\n".join(context_parts) if context_parts else "No matching data was found in ClinicalTrials.gov or FAERS for this query."

    if not context_parts:
        answer = "I did not find any matching trials in the available data."
        return {
            **state,
            "answer": answer,
            "citations": [],
            "usage": state.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        }

    resp_text, usage_delta = llm_client.complete(
        SYNTH_SYSTEM, f"Context:\n{context}\n\nQuestion: {state['query']}", max_tokens=800
    )
    answer = resp_text.strip()

    usage = state.get("usage", {"input_tokens": 0, "output_tokens": 0})
    usage["input_tokens"] += usage_delta["input_tokens"]
    usage["output_tokens"] += usage_delta["output_tokens"]

    return {
        **state,
        "answer": answer,
        "citations": sorted(set(citations)),
        "usage": usage,
    }


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("structured_lookup", structured_lookup)
    graph.add_node("semantic_lookup", semantic_lookup)
    graph.add_node("adverse_event_lookup", adverse_event_lookup)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("extract_entities")
    graph.add_edge("extract_entities", "structured_lookup")
    graph.add_edge("structured_lookup", "semantic_lookup")
    graph.add_edge("semantic_lookup", "adverse_event_lookup")
    graph.add_edge("adverse_event_lookup", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(query: str) -> AgentState:
    graph = get_graph()
    start = time.time()
    result = graph.invoke({"query": query, "usage": {"input_tokens": 0, "output_tokens": 0}})
    result["latency_ms"] = int((time.time() - start) * 1000)
    return result