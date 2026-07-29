"""
Real-time ingestion from two public, keyless APIs:
  1. ClinicalTrials.gov API v2 - structured trial metadata
  2. openFDA FAERS - real-world adverse event reports

Both are called live at query time (with a short cache in SQLite) rather than
bulk-downloaded, so results reflect the current state of the registry.

IMPORTANT: this uses the `requests` library, not `httpx`. ClinicalTrials.gov's
WAF blocks httpx's TLS fingerprint specifically and returns 403 regardless of
headers. requests is not blocked. Do not swap this back to httpx.
"""
import requests
from datetime import datetime
from backend.config import CLINICALTRIALS_BASE, OPENFDA_BASE, MAX_INGEST_RESULTS
from backend.db import get_session, Trial, AdverseEvent

HTTP_TIMEOUT = 20.0

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}

FIELDS = ",".join([
    "NCTId", "BriefTitle", "Condition", "Phase", "OverallStatus",
    "InterventionName", "EligibilityCriteria", "BriefSummary",
    "LeadSponsorName", "StartDate", "LastUpdatePostDate",
])


def _parse_studies(data: dict) -> list[dict]:
    studies = data.get("studies", [])
    parsed = []
    for s in studies:
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        cond_mod = proto.get("conditionsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        elig = proto.get("eligibilityModule", {})
        desc = proto.get("descriptionModule", {})
        sponsor = proto.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})

        interventions = arms.get("interventions", [])
        intervention_str = "; ".join(i.get("name", "") for i in interventions)

        parsed.append({
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "condition": "; ".join(cond_mod.get("conditions", [])),
            "phase": "; ".join(design.get("phases", [])) or "N/A",
            "status": status.get("overallStatus", ""),
            "intervention": intervention_str,
            "eligibility_criteria": elig.get("eligibilityCriteria", ""),
            "brief_summary": desc.get("briefSummary", ""),
            "sponsor": sponsor.get("name", ""),
            "start_date": status.get("startDateStruct", {}).get("date", ""),
            "last_update": status.get("lastUpdatePostDateStruct", {}).get("date", ""),
        })
    return parsed


def fetch_trials(condition: str, max_results: int = None) -> list[dict]:
    """
    Query ClinicalTrials.gov v2 for studies matching a condition/drug term.

    Tries query.cond first (matches the registry's structured Condition
    field). If that returns nothing -- which happens for abbreviations like
    "NASH" that don't always match the structured field -- falls back to
    query.term, which does a full-text search across all fields including
    titles and descriptions, so abbreviations and drug names still hit.
    """
    max_results = max_results or MAX_INGEST_RESULTS
    base_params = {
        "pageSize": min(max_results, 100),
        "fields": FIELDS,
    }

    for param_name in ("query.cond", "query.term"):
        params = {**base_params, param_name: condition}
        resp = requests.get(CLINICALTRIALS_BASE, params=params, headers=REQUEST_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        parsed = _parse_studies(data)
        print(f"[ingest.fetch_trials] '{condition}' via {param_name} -> {len(parsed)} studies")
        if parsed:
            return parsed

    return []


def persist_trials(trials: list[dict]) -> int:
    session = get_session()
    count = 0
    try:
        for t in trials:
            if not t["nct_id"]:
                continue
            existing = session.get(Trial, t["nct_id"])
            if existing:
                for k, v in t.items():
                    setattr(existing, k, v)
                existing.fetched_at = datetime.utcnow()
            else:
                session.add(Trial(**t))
                count += 1
        session.commit()
    finally:
        session.close()
    return count


def fetch_adverse_events(drug_name: str, limit: int = 10) -> list[dict]:
    """Query openFDA FAERS for the most commonly reported reactions for a drug."""
    params = {
        "search": f'patient.drug.medicinalproduct:"{drug_name}"',
        "count": "patient.reaction.reactionmeddrapt.exact",
        "limit": limit,
    }
    resp = requests.get(OPENFDA_BASE, params=params, headers=REQUEST_HEADERS, timeout=HTTP_TIMEOUT)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    parsed = [{"drug_name": drug_name, "reaction": r["term"], "report_count": r["count"]} for r in results]

    session = get_session()
    try:
        for r in parsed:
            session.add(AdverseEvent(**r))
        session.commit()
    finally:
        session.close()
    return parsed