from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.db import init_db, get_session, QueryLog
from backend.agents import run_query

# Groq's free tier has no per-token cost. These are set to 0 by default so the
# /api/metrics cost field is accurate; if you switch to a paid provider,
# update these (USD per million tokens) to reflect real pricing so the LLMOps
# cost tracking stays meaningful.
PRICE_PER_M_INPUT = 0.0
PRICE_PER_M_OUTPUT = 0.0

app = FastAPI(title="TrialLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    route_info: dict
    latency_ms: int
    estimated_cost_usd: float


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = run_query(req.query)
    usage = result.get("usage", {"input_tokens": 0, "output_tokens": 0})
    cost = (usage["input_tokens"] / 1_000_000) * PRICE_PER_M_INPUT + (
        usage["output_tokens"] / 1_000_000
    ) * PRICE_PER_M_OUTPUT

    session = get_session()
    try:
        session.add(
            QueryLog(
                query=req.query,
                route=f"trials={len(result.get('trials', []))},chunks={len(result.get('semantic_chunks', []))},ae={len(result.get('adverse_events', []))}",
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                estimated_cost_usd=cost,
                latency_ms=result.get("latency_ms", 0),
                answer_preview=result["answer"][:200],
            )
        )
        session.commit()
    finally:
        session.close()

    return ChatResponse(
        answer=result["answer"],
        citations=result.get("citations", []),
        route_info={
            "trials_found": len(result.get("trials", [])),
            "semantic_chunks_found": len(result.get("semantic_chunks", [])),
            "adverse_event_records": len(result.get("adverse_events", [])),
        },
        latency_ms=result.get("latency_ms", 0),
        estimated_cost_usd=round(cost, 6),
    )


@app.get("/api/metrics")
def metrics():
    """Lightweight LLMOps dashboard data: volume, latency, cost over recent queries."""
    session = get_session()
    try:
        logs = session.query(QueryLog).order_by(QueryLog.id.desc()).limit(100).all()
    finally:
        session.close()

    if not logs:
        return JSONResponse({"total_queries": 0})

    total_cost = sum(l.estimated_cost_usd or 0 for l in logs)
    avg_latency = sum(l.latency_ms or 0 for l in logs) / len(logs)

    return JSONResponse(
        {
            "total_queries": len(logs),
            "total_estimated_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "recent": [
                {
                    "query": l.query,
                    "route": l.route,
                    "latency_ms": l.latency_ms,
                    "cost_usd": round(l.estimated_cost_usd or 0, 6),
                    "created_at": l.created_at.isoformat() if l.created_at else None,
                }
                for l in logs[:20]
            ],
        }
    )


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
