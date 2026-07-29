import os
from dotenv import load_dotenv

load_dotenv()

# LLM provider config -- OpenAI-compatible endpoint, defaults to Groq's free tier.
# To switch providers, just change these three values (see README "Free LLM providers").
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

DATABASE_PATH = os.getenv("DATABASE_PATH", "./triallens.db")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
MAX_INGEST_RESULTS = int(os.getenv("MAX_INGEST_RESULTS", "50"))

CLINICALTRIALS_BASE = "https://clinicaltrials.gov/api/v2/studies"
OPENFDA_BASE = "https://api.fda.gov/drug/event.json"
