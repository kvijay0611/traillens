# TrialLens — Clinical Trial Intelligence Copilot

A multi-agent RAG system that answers natural-language questions about clinical trials by combining:
- **Live structured data** from ClinicalTrials.gov (trial phase, status, sponsor, eligibility)
- **Semantic search** over trial descriptions and eligibility criteria (RAG)
- **Real-world adverse event data** from FDA FAERS (openFDA)

Ask things like *"What Phase 3 trials exist for NASH?"* or *"What adverse events have been reported for semaglutide?"* and get a synthesized, cited answer.

Runs on **Groq's free tier** by default (fast, no cost) — swappable to any OpenAI-compatible provider with a 3-line config change.

---

## Setup from scratch

### 1. Get the code onto your machine

Unzip the project, then open a terminal in the `triallens/` folder.

### 2. Install Python 3.11+

Check your version:
```bash
python3 --version
```
If you don't have 3.11+, install it from [python.org](https://www.python.org/downloads/).

### 3. Create a virtual environment

```bash
python3 -m venv venv

# Activate it:
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```
You'll know it worked because your terminal prompt will show `(venv)`.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```
This takes a couple of minutes — it's pulling in FastAPI, LangGraph, Chroma, and sentence-transformers (which includes a small ML library for embeddings).

### 5. Get a free Groq API key

1. Go to [console.groq.com/keys](https://console.groq.com/keys)
2. Sign up (free, no credit card required)
3. Click **Create API Key**, copy it — it starts with `gsk_`

Groq's free tier gives you generous rate limits on fast Llama models. No card, no trial expiry.

### 6. Configure your environment

```bash
cp .env.example .env
```
Open `.env` in any text editor and paste your key:
```
LLM_API_KEY=gsk_your_actual_key_here
```
Leave `LLM_BASE_URL` and `LLM_MODEL` as-is — they're already set for Groq.

### 7. Run it locally

```bash
uvicorn backend.main:app --reload
```
Open **http://localhost:8000** in your browser. Try one of the example question chips.

First query will be slower (downloading the embedding model + hitting live APIs the first time); after that it's fast, and repeated questions about the same condition are cached in a local SQLite file.

### 8. Run the eval harness (optional but recommended before deploying)

```bash
python -m eval.run_eval
```
This runs 8 test questions through the full pipeline and writes a score report to `eval/eval_report.json`.

---

## Using a different free LLM provider

The app talks to any OpenAI-compatible `/chat/completions` endpoint, so switching providers is just editing 3 lines in `.env`:

| Provider | LLM_BASE_URL | Example LLM_MODEL | Notes |
|---|---|---|---|
| **Groq** (default) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` | Free, very fast |
| Cerebras | `https://api.cerebras.ai/v1` | `llama-3.3-70b` | Free tier, extremely fast |
| Together AI | `https://api.together.xyz/v1` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Free starter credits |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.3-70b-instruct:free` | Many free-tagged models |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-2.0-flash` | Free tier via OpenAI-compat endpoint |

No code changes needed — `backend/llm_client.py` is the only place that talks to the LLM, and it's provider-agnostic by design. This is worth mentioning in interviews: the app isn't hard-coupled to one vendor.

---

## Architecture

```
User query
   |
   v
[extract_entities]     <- LLM parses query into {condition, drug, phase, wants_adverse_events}
   |
   v
[structured_lookup]    <- SQLite query; auto-ingests fresh data from ClinicalTrials.gov if thin
   |
   v
[semantic_lookup]      <- Chroma vector search over trial summaries/eligibility text
   |
   v
[adverse_event_lookup] <- openFDA FAERS lookup if a drug + safety intent was detected
   |
   v
[synthesize]           <- LLM combines all context into a cited answer
   |
   v
Answer + citations (NCT IDs) + latency + cost, logged to query_logs table
```

This is a **LangGraph** state machine, not a single prompt — each node only runs the work it needs (a pure semantic question skips the FAERS API call, etc.), and every step is logged for observability.

### Why these design choices
- **SQLite + Chroma (file-based), not Postgres + Pinecone** — keeps this deployable as a single service with zero extra infra, which matters for a solo, fast-shipping portfolio piece. Swapping `backend/db.py` for Postgres or `backend/vectorstore.py` for pgvector is a small, contained change — a good "designed for it, didn't need it yet" talking point.
- **Structured DB + vector search kept separate** — avoids hallucinating precise facts (trial phase, status) that should come from exact lookups, not embeddings.
- **Provider-agnostic LLM layer** — `backend/llm_client.py` isolates the OpenAI-compatible call so swapping Groq for Cerebras, OpenRouter, or a paid provider later is a config change, not a rewrite.
- **LLMOps logging table** — every query logs input/output tokens, estimated cost, latency, and which agent branches fired, viewable at `/api/metrics`.

---

## Eval Harness

`eval/golden_dataset.json` has 8 hand-curated test cases spanning structured lookups, semantic questions, adverse-event questions, and two **negative cases** (an out-of-domain question and a made-up condition) to check the system doesn't fabricate answers.

It scores:
1. **Keyword recall** — do expected terms appear in the answer
2. **Citation check** — does the answer cite an NCT ID when it should
3. **Refusal check** — does the system correctly decline to answer when there's no data, instead of hallucinating

This is the piece most portfolio RAG projects skip — mention it explicitly in interviews and resumes, and quote your actual `eval_report.json` numbers once you've run it.

### Extending the eval harness
For a more rigorous version, swap the keyword-matching for [RAGAS](https://github.com/explodinggradients/ragas) metrics (faithfulness, context precision/recall) using your LLM as the judge model. The golden dataset format is already compatible.

---

## Deploying it (Render, free tier, ~10 minutes)

1. **Push this folder to a new GitHub repo:**
   ```bash
   cd triallens
   git init
   git add .
   git commit -m "Initial commit: TrialLens"
   git remote add origin https://github.com/<your-username>/triallens.git
   git push -u origin main
   ```

2. **Create a Render account** at [render.com](https://render.com) (free, GitHub sign-in is fastest).

3. **New → Blueprint** → connect your `triallens` repo. Render detects `render.yaml` automatically and provisions the service.
   - Manual alternative: **New → Web Service** → connect repo → Environment: **Docker** → it picks up the `Dockerfile`.

4. **Set the environment variable** `LLM_API_KEY` in the Render dashboard (Environment tab) with your Groq key — it's marked `sync: false` in `render.yaml` so it's never committed to git.

5. **Deploy.** First build takes 5–8 minutes. Render gives you a live URL like `https://triallens.onrender.com`.

> Free tier note: Render's free web services spin down after inactivity and take ~30–60s to wake on the next request. Fine for a portfolio demo; upgrade to a paid instance if you want it always-warm for an interview screen-share.

### Alternative: Hugging Face Spaces
1. Create a new Space, SDK: **Docker**.
2. Push this repo's contents to the Space's git remote.
3. Add `LLM_API_KEY` as a Space secret (Settings → Repository secrets).
4. The existing `Dockerfile` works as-is.

---

## Scaling this up (talking points for interviews)

- **Vector store**: swap Chroma for pgvector on managed Postgres (Neon/Supabase) for multi-instance deployment or larger corpora — `vectorstore.py` isolates this behind a small interface.
- **Observability**: swap the custom `query_logs` table for [Langfuse](https://langfuse.com) (self-hostable, free tier) for full trace visualization per agent node.
- **Structured extraction**: `extract_entities` currently parses free-form JSON from the model; production would use guaranteed structured outputs / tool calling instead of text parsing.
- **Caching**: add Redis in front of `fetch_trials`/`fetch_adverse_events` to avoid re-hitting external APIs for popular conditions.
- **Multi-tenancy**: the SQLite file is fine for a single-user demo; a real deployment would need per-user session isolation or a shared Postgres instance.

---

## Project structure

```
triallens/
├── backend/
│   ├── main.py          # FastAPI app, /api/chat and /api/metrics endpoints
│   ├── agents.py         # LangGraph multi-agent pipeline
│   ├── llm_client.py     # Provider-agnostic LLM wrapper (Groq by default)
│   ├── ingest.py          # ClinicalTrials.gov + openFDA API clients
│   ├── vectorstore.py    # Chroma RAG layer
│   ├── db.py              # SQLAlchemy models (Trial, AdverseEvent, QueryLog)
│   └── config.py
├── eval/
│   ├── golden_dataset.json
│   └── run_eval.py
├── frontend/
│   └── index.html         # Chat UI, no build step needed
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## Resume-ready description

> Built TrialLens, a multi-agent RAG system (LangGraph, Chroma, Groq/Llama) synthesizing clinical trial data from live ClinicalTrials.gov and FDA FAERS APIs; built a golden-dataset eval harness (8+ cases including adversarial no-data/out-of-domain checks) scoring keyword recall, citation accuracy, and refusal correctness; instrumented full LLMOps logging (token usage, latency per query) exposed via a metrics API; designed a provider-agnostic LLM layer supporting Groq, Cerebras, and OpenRouter; deployed as a single containerized service on Render.

Fill in real eval numbers once you've run `eval/run_eval.py` against your own deployment — real, reproducible metrics from your own run are far more credible in an interview than placeholder numbers.
