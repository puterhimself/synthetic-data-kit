# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Text analysis utilities for extracting insights from text
import re
import time
from typing import List, Optional, Dict, Tuple
from collections import Counter
from pathlib import Path


# Common English stopwords
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had',
    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'what', 'which', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don',
    'should', 'now', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further',
    'then', 'once'
}


def normalize_text(text: str) -> str:
    """Normalize text for analysis"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)
    return text.strip()


def extract_text_from_parser_result(parser_result) -> str:
    """Extract plain text from parser result (handles both dict and string formats)"""
    if isinstance(parser_result, str):
        return parser_result
    elif isinstance(parser_result, list):
        # Handle list of dicts (common parser format)
        texts = []
        for item in parser_result:
            if isinstance(item, dict):
                if 'text' in item:
                    texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        return '\n\n'.join(texts)
    elif isinstance(parser_result, dict):
        if 'text' in parser_result:
            return parser_result['text']
    return str(parser_result)


def estimate_token_count(text: str) -> int:
    """Estimate token count (rough approximation: ~4 chars per token)"""
    return len(text) // 4


def generate_summary(text: str, max_sentences: int = 3) -> str:
    """Generate a simple summary using lead sentences"""
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return ""
    
    # Take first N sentences that are substantial
    summary_sentences = []
    for sentence in sentences:
        if len(sentence) > 20:  # Filter out very short sentences
            summary_sentences.append(sentence)
            if len(summary_sentences) >= max_sentences:
                break
    
    if not summary_sentences:
        # Fallback: use first sentence regardless of length
        summary_sentences = [sentences[0]] if sentences else []
    
    return '. '.join(summary_sentences) + '.' if summary_sentences else ""


def extract_keywords(text: str, top_k: int = 15, min_length: int = 3) -> List[str]:
    """Extract top keywords from text using frequency analysis"""
    # Normalize text
    normalized = normalize_text(text.lower())
    
    # Extract words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', normalized)
    
    # Filter out stopwords and short words
    keywords = [w for w in words if w not in STOPWORDS and len(w) >= min_length]
    
    # Count frequencies
    word_freq = Counter(keywords)
    
    # Get top K keywords
    top_keywords = [word for word, _ in word_freq.most_common(top_k)]
    
    return top_keywords


def extract_tags(text: str, keywords: List[str]) -> List[str]:
    """Extract tags from keywords (can be enhanced with domain-specific logic)"""
    # For now, just use top keywords as tags
    # Could be enhanced with semantic grouping or domain knowledge
    return keywords[:10]  # Limit tags to top 10


def categorize_file(file_path: Path, text: str, keywords: List[str]) -> Optional[str]:
    """Categorize file based on format and content"""
    ext = file_path.suffix.lower()
    text_lower = text.lower()
    
    # Format-based categorization
    format_map = {
        '.pdf': 'document',
        '.docx': 'document',
        '.pptx': 'presentation',
        '.txt': 'text',
        '.html': 'webpage',
        '.htm': 'webpage',
    }
    
    category = format_map.get(ext, 'unknown')
    
    # Content-based refinement
    if any(kw in text_lower for kw in ['invoice', 'receipt', 'payment', 'bill']):
        return 'invoice'
    elif any(kw in text_lower for kw in ['slide', 'presentation', 'deck']):
        return 'presentation'
    elif any(kw in text_lower for kw in ['code', 'function', 'class', 'import', 'def ']):
        return 'code'
    elif any(kw in text_lower for kw in ['policy', 'terms', 'agreement', 'legal']):
        return 'policy'
    elif any(kw in text_lower for kw in ['email', 'mail', '@']):
        return 'email'
    elif any(kw in text_lower for kw in ['question', 'answer', 'qa', 'quiz']):
        return 'qa'
    
    return category


def detect_language(text: str) -> Optional[str]:
    """Simple language detection based on common words"""
    text_lower = text.lower()
    
    # Common words in different languages (very basic heuristic)
    english_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
    spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se']
    french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir']
    
    english_count = sum(1 for word in english_words if word in text_lower)
    spanish_count = sum(1 for word in spanish_words if word in text_lower)
    french_count = sum(1 for word in french_words if word in text_lower)
    
    if english_count > spanish_count and english_count > french_count:
        return 'en'
    elif spanish_count > french_count:
        return 'es'
    elif french_count > 0:
        return 'fr'
    
    # Default to English if we can't determine
    return 'en'


def detect_pii(text: str) -> List[str]:
    """Detect potential PII using regex patterns"""
    warnings = []
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if re.search(email_pattern, text):
        warnings.append('potential_email')
    
    # Phone number pattern (US format)
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    if re.search(phone_pattern, text):
        warnings.append('potential_phone')
    
    # SSN pattern (US format)
    ssn_pattern = r'\b\d{3}-?\d{2}-?\d{4}\b'
    if re.search(ssn_pattern, text):
        warnings.append('potential_ssn')
    
    # Credit card pattern (simplified)
    cc_pattern = r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
    if re.search(cc_pattern, text):
        warnings.append('potential_cc')
    
    return warnings


def analyze_text_quality(text: str) -> List[str]:
    """Analyze text quality and return warnings"""
    warnings = []
    
    if len(text) < 50:
        warnings.append('very_short')
    
    # Check symbol ratio
    if len(text) > 0:
        symbol_ratio = len(re.findall(r'[^\w\s]', text)) / len(text)
        if symbol_ratio > 0.3:
            warnings.append('high_symbol_ratio')
    
    # Check for mostly whitespace
    if len(text.strip()) < len(text) * 0.5:
        warnings.append('mostly_whitespace')
    
    return warnings

