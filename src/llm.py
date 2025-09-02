# src/llm.py
import os
import openai
from .config import OPENAI_API_KEY, OPENAI_MODEL

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

SYSTEM_PROMPT_VERIFIER = (
    "You are an objective fact-check assistant. For each claim, answer with a short verdict: "
    "'SUPPORTED', 'NOT_SUPPORTED', or 'UNCLEAR'. Provide a concise reasoning (1-2 sentences) and up to 3 evidence links. "
    "If you cannot find reliable evidence, say 'UNCLEAR'. Be concise and include confidence (0-1)."
)

def verify_claim_with_llm(claim_text: str, context: str = "") -> dict:
    """Ask LLM to verify the claim. Returns dict: {verdict, confidence, reasoning, evidence: [{title,url}], raw}"""
    if not OPENAI_API_KEY:
        return {"verdict": "NO_API", "confidence": 0.0, "reasoning": "OpenAI key missing.", "evidence": []}
    prompt = f"Claim: {claim_text}\n\nContext/Article Excerpt (optional):\n{context}\n\nTask: Verify the claim and provide verdict, confidence (0-1), short reasoning, and up to 3 evidence links if available."
    try:
        resp = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_VERIFIER},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=400,
        )
        txt = resp["choices"][0]["message"]["content"]
        return {"raw": txt, "verdict_text": txt}
    except Exception as e:
        return {"verdict": "ERROR", "confidence": 0.0, "reasoning": str(e), "evidence": []}
