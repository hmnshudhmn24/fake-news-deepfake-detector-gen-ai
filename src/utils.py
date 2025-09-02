# src/utils.py
import re
import requests
from bs4 import BeautifulSoup
from newspaper import Article

def fetch_article_text_from_url(url: str) -> str:
    try:
        a = Article(url)
        a.download()
        a.parse()
        return a.text
    except Exception:
        try:
            r = requests.get(url, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            paragraphs = soup.find_all("p")
            return "\n\n".join([p.get_text() for p in paragraphs])
        except Exception:
            return ""

def split_into_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def extract_candidate_claims(article_text: str, max_claims: int = 6):
    sents = split_into_sentences(article_text)
    candidates = []
    for s in sents:
        if re.search(r'\b(%)|\b\d{4}\b|\b\d+\b|\b(according to|reports|said|claimed|alleges)\b', s, re.I):
            candidates.append(s)
        if len(candidates) >= max_claims:
            break
    if not candidates:
        candidates = sents[:max_claims]
    return candidates
