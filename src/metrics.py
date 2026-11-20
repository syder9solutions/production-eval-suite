"""
Evaluation Metrics Engine with Stopword-Filtered Factual Attribution.
Computes:
- Faithfulness (claim attribution against reference context)
- Answer Relevance (prompt coverage)
- Refusal Accuracy (proper rejection of out-of-domain prompts)
- Hallucination Rate (unsupported assertions)
"""
import math
import re

STOPWORDS = {
    "the", "and", "are", "for", "with", "this", "that", "from", "into", "between",
    "across", "all", "its", "our", "your", "they", "been", "have", "were", "what",
    "when", "where", "which", "will", "would", "shall", "should", "does", "done"
}


def clean_content_words(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS}


def evaluate_faithfulness(answer: str, context: str) -> float:
    """Computes proportion of factual content in answer supported by the reference context."""
    if "cannot answer this question based on the provided documentation" in answer.lower():
        return 1.0  # Legitimate refusal is 100% faithful

    claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not claims:
        return 1.0

    context_words = clean_content_words(context)
    if not context_words:
        return 0.0

    claim_scores = []
    for claim in claims:
        claim_words = clean_content_words(claim)
        if not claim_words:
            continue
        overlap = claim_words.intersection(context_words)
        ratio = len(overlap) / float(len(claim_words))
        claim_scores.append(ratio)

    if not claim_scores:
        return 1.0

    mean_support = sum(claim_scores) / float(len(claim_scores))
    return round(mean_support, 3)


def evaluate_answer_relevance(prompt: str, answer: str) -> float:
    """Measures topical overlap between the user prompt and the generated response."""
    prompt_words = clean_content_words(prompt)
    if not prompt_words:
        return 1.0

    answer_words = clean_content_words(answer)
    overlap = prompt_words.intersection(answer_words)
    return round(len(overlap) / float(len(prompt_words)), 3)


def is_refusal(answer: str) -> bool:
    return "cannot answer this question based on the provided documentation" in answer.lower()


def compute_hallucination_rate(faithfulness_score: float) -> float:
    return round(max(0.0, 1.0 - faithfulness_score), 3)
