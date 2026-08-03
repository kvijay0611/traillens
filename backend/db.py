from sqlalchemy import create_engine, Column, String, Integer, Text, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from backend.config import DATABASE_PATH

engine = create_engine(f"sqlite:///{DATABASE_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Trial(Base):
    """A single clinical trial pulled from ClinicalTrials.gov."""
    __tablename__ = "trials"

    nct_id = Column(String, primary_key=True)
    title = Column(Text)
    condition = Column(Text)
    phase = Column(String)
    status = Column(String)
    intervention = Column(Text)
    eligibility_criteria = Column(Text)
    brief_summary = Column(Text)
    sponsor = Column(Text)
    start_date = Column(String)
    last_update = Column(String)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class TrialChunk(Base):
    """
    Text chunks (summary + eligibility segments) used for lightweight
    keyword-overlap retrieval -- see backend/vectorstore.py for why this
    replaced a Chroma/embedding-model based approach.
    """
    __tablename__ = "trial_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nct_id = Column(String, index=True)
    section = Column(String)
    text = Column(Text)
    title = Column(Text)
    condition = Column(Text)
    phase = Column(String)
    status = Column(String)


class AdverseEvent(Base):
    """Aggregated adverse-event counts from openFDA FAERS, per drug."""
    __tablename__ = "adverse_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    drug_name = Column(String, index=True)
    reaction = Column(String)
    report_count = Column(Integer)
    fetched_at = Column(DateTime, default=datetime.utcnow)


class QueryLog(Base):
    """LLMOps log: one row per user query, for observability + cost tracking."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text)
    route = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    estimated_cost_usd = Column(Float)
    latency_ms = Column(Integer)
    answer_preview = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    return SessionLocal()
