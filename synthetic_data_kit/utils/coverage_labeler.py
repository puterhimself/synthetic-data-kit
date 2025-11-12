# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Coverage labeling: keyword rules first, LLM fallback for low-confidence/unlabeled

from typing import Dict, List, Optional, Any
import re
from collections import Counter


def label_coverage_keyword_rules(
    text: str, keywords: List[str], keyword_rules: Optional[Dict[str, List[str]]] = None
) -> Optional[str]:
    """Label coverage using keyword rules.

    Args:
        text: Text to analyze
        keywords: Extracted keywords from the text
        keyword_rules: Domain-specific keyword rules mapping domain -> keywords

    Returns:
        Coverage label (domain/topic) or None if no match
    """
    if not keyword_rules:
        return None

    text_lower = text.lower()
    keyword_lower = [k.lower() for k in keywords]

    # Score each domain based on keyword matches
    domain_scores = {}
    for domain, domain_keywords in keyword_rules.items():
        score = 0
        # Check text for domain keywords
        for kw in domain_keywords:
            if kw.lower() in text_lower:
                score += 2  # Full match in text
            if kw.lower() in keyword_lower:
                score += 1  # Match in extracted keywords

        if score > 0:
            domain_scores[domain] = score

    if domain_scores:
        # Return domain with highest score
        return max(domain_scores.items(), key=lambda x: x[1])[0]

    return None


def label_coverage_llm_fallback(
    text: str,
    keywords: List[str],
    llm_client: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Label coverage using LLM fallback when keyword rules don't match.

    Args:
        text: Text to analyze
        keywords: Extracted keywords
        llm_client: LLM client for analysis
        config: Configuration dict

    Returns:
        Coverage label or None
    """
    if not llm_client:
        return None

    # Use a simple prompt to classify the text's domain/topic
    prompt = f"""Analyze this text and determine its primary domain or topic.

Text snippet: {text[:500]}
Keywords: {", ".join(keywords[:10])}

Respond with a single domain/topic label (e.g., "technical", "general", "academic", "business", "medical", etc.) or "unknown" if unclear.

Domain/Topic:"""

    try:
        messages = [
            {"role": "system", "content": "You are a text classification assistant."},
            {"role": "user", "content": prompt},
        ]
        response = llm_client.chat_completion(messages, temperature=0.1)
        label = response.strip().lower()
        # Clean up common prefixes/suffixes
        label = re.sub(r"^(domain|topic|label):\s*", "", label)
        label = label.strip("\"'")
        return label if label != "unknown" else None
    except Exception:
        return None


def label_coverage(
    text: str,
    keywords: List[str],
    keyword_rules: Optional[Dict[str, List[str]]] = None,
    llm_fallback: bool = True,
    llm_client: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Hybrid coverage labeler: keyword rules first, LLM fallback.

    Args:
        text: Text to analyze
        keywords: Extracted keywords
        keyword_rules: Domain-specific keyword rules
        llm_fallback: Whether to use LLM if keyword rules don't match
        llm_client: LLM client for fallback
        config: Configuration dict

    Returns:
        Coverage label or None
    """
    # Try keyword rules first
    label = label_coverage_keyword_rules(text, keywords, keyword_rules)

    # If no match and LLM fallback enabled, try LLM
    if label is None and llm_fallback:
        label = label_coverage_llm_fallback(text, keywords, llm_client, config)

    return label
