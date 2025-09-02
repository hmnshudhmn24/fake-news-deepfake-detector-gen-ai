# src/article_checker.py
from typing import List, Dict, Optional
from .utils import fetch_article_text_from_url, extract_candidate_claims
from .llm import verify_claim_with_llm
from .config import MAX_CLAIMS

def analyze_article(text: str = "", url: Optional[str] = None, max_claims: int = None) -> Dict:
    if max_claims is None:
        max_claims = MAX_CLAIMS

    if url and not text:
        text = fetch_article_text_from_url(url)

    if not text:
        return {"article_verdict": "NO_TEXT", "claims": [], "llm_summary": "No article text provided."}

    candidate_claims = extract_candidate_claims(text, max_claims)
    claims_results = []
    support_count = 0
    not_support_count = 0
    unclear_count = 0

    for c in candidate_claims:
        resp = verify_claim_with_llm(c, context=text[:2000])
        raw = resp.get("raw") or resp.get("verdict_text") or ""
        lower = raw.lower()
        if "supported" in lower:
            verdict = "SUPPORTED"
            support_count += 1
        elif "not_supported" in lower or "not supported" in lower or "contradicted" in lower:
            verdict = "NOT_SUPPORTED"
            not_support_count += 1
        elif "unclear" in lower or "insufficient" in lower:
            verdict = "UNCLEAR"
            unclear_count += 1
        else:
            verdict = "UNCLEAR"

        claims_results.append({
            "claim_text": c,
            "verdict": verdict,
            "raw": raw,
            "llm_response": raw
        })

    total = len(claims_results) or 1
    if not_support_count / total > 0.5:
        article_verdict = "LIKELY_FALSE"
    elif support_count / total > 0.6:
        article_verdict = "LIKELY_TRUE"
    elif unclear_count / total > 0.5:
        article_verdict = "UNCLEAR"
    else:
        article_verdict = "MIXED"

    return {
        "article_verdict": article_verdict,
        "claims": claims_results,
        "llm_summary": f"Claims analyzed: {total}, supported: {support_count}, not_supported: {not_support_count}, unclear: {unclear_count}"
    }
