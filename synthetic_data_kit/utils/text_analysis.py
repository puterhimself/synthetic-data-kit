# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Text analysis utilities for extracting insights from text using spacy and nltk
import re
import time
from typing import List, Optional, Dict, Tuple, Union
from collections import Counter
from pathlib import Path
import logging

import spacy
from spacy.language import Language
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

logger = logging.getLogger(__name__)

# Global NLP models - lazy loaded
_nlp_model: Optional[Language] = None
_lemmatizer: Optional[WordNetLemmatizer] = None
_stopwords_set: Optional[set] = None


def _ensure_nltk_data() -> Optional[str]:
    """Ensure required NLTK data is downloaded. Returns error string if failed, None on success."""
    required_data = ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger"]
    for data in required_data:
        nltk_result = nltk.download(data, quiet=True)
        if not nltk_result:
            return f"Failed to download NLTK data: {data}"
    return None


def _load_spacy_model() -> Union[Language, str]:
    """Load spacy model. Returns model on success, error string on failure."""
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    model_name = "en_core_web_sm"
    if not spacy.util.is_package(model_name):
        return f"Spacy model '{model_name}' not found. Please install it with: python -m spacy download {model_name}"

    _nlp_model = spacy.load(model_name)
    return _nlp_model


def _get_lemmatizer() -> WordNetLemmatizer:
    """Get NLTK lemmatizer instance."""
    global _lemmatizer
    if _lemmatizer is None:
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def _get_stopwords() -> set:
    """Get NLTK stopwords set."""
    global _stopwords_set
    if _stopwords_set is None:
        _stopwords_set = set(stopwords.words("english"))
    return _stopwords_set


def initialize_nlp_models() -> Dict[str, Optional[str]]:
    """Initialize all NLP models and return status.

    Returns:
        Dictionary with initialization status for each model.
        Keys: 'nltk', 'spacy', 'lemmatizer', 'stopwords'
        Values: None on success, error string on failure
    """
    status = {}

    # Initialize NLTK data
    nltk_error = _ensure_nltk_data()
    status["nltk"] = nltk_error
    if not nltk_error:
        logger.info("NLTK data initialized successfully")
    else:
        logger.warning(f"NLTK initialization issue: {nltk_error}")

    # Initialize spacy model
    nlp = _load_spacy_model()
    if isinstance(nlp, str):
        status["spacy"] = nlp
        logger.warning(f"Spacy initialization issue: {nlp}")
    else:
        status["spacy"] = None
        logger.info("Spacy model loaded successfully")

    # Initialize lemmatizer
    lemmatizer = _get_lemmatizer()
    status["lemmatizer"] = None if lemmatizer else "Failed to load lemmatizer"

    # Initialize stopwords
    stopwords_error = None
    stopwords_set = _get_stopwords()
    if not stopwords_set:
        stopwords_error = "Failed to load stopwords"
    status["stopwords"] = stopwords_error

    return status


def normalize_text(text: str) -> str:
    """Normalize text for analysis using spacy. Falls back to regex on error."""
    nlp = _load_spacy_model()
    if isinstance(nlp, str):
        # Fallback to regex normalization
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.\,\!\?\;\:\-]", "", text)
        return text.strip()

    # Use spacy for normalization
    doc = nlp(text[:100000])  # Limit text size for performance
    normalized_tokens = []
    for token in doc:
        if not token.is_space:
            normalized_tokens.append(token.text)
    return " ".join(normalized_tokens)


def extract_text_from_parser_result(parser_result) -> str:
    """Extract plain text from parser result (handles both dict and string formats)"""
    if isinstance(parser_result, str):
        return parser_result
    elif isinstance(parser_result, list):
        # Handle list of dicts (common parser format)
        texts = []
        for item in parser_result:
            if isinstance(item, dict):
                if "text" in item:
                    texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return "\n\n".join(texts)
    elif isinstance(parser_result, dict):
        if "text" in parser_result:
            return parser_result["text"]
    return str(parser_result)


def estimate_token_count(text: str) -> int:
    """Estimate token count using spacy tokenization. Falls back to char-based on error."""
    nlp = _load_spacy_model()
    if isinstance(nlp, str):
        # Fallback: rough approximation ~4 chars per token
        return len(text) // 4

    # Use spacy tokenization
    doc = nlp(text[:100000])  # Limit text size for performance
    return len([token for token in doc if not token.is_space])


def generate_summary(text: str, max_sentences: int = 3) -> str:
    """Generate a simple summary using lead sentences with nltk sentence tokenization."""
    nltk_error = _ensure_nltk_data()
    if nltk_error:
        # Fallback to regex
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
    else:
        # Use NLTK for better sentence tokenization
        sentences = sent_tokenize(text)

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

    return " ".join(summary_sentences)


def extract_keywords(text: str, top_k: int = 15, min_length: int = 3) -> List[str]:
    """Extract top keywords using spacy NLP (lemmatization, POS tagging) and NLTK stopwords."""
    nlp = _load_spacy_model()
    nltk_error = _ensure_nltk_data()

    if isinstance(nlp, str) or nltk_error:
        # Fallback to simple frequency analysis
        normalized = text.lower()
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        words = re.findall(r"\b\w+\b", normalized)

        # Simple stopword set as fallback
        simple_stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
        }
        keywords = [w for w in words if w not in simple_stopwords and len(w) >= min_length]
        word_freq = Counter(keywords)
        return [word for word, _ in word_freq.most_common(top_k)]

    # Use spacy for advanced keyword extraction
    doc = nlp(text[:100000])  # Limit text size
    stopwords_set = _get_stopwords()

    # Extract lemmatized nouns, proper nouns, and adjectives
    keywords = []
    for token in doc:
        if (
            token.pos_ in ["NOUN", "PROPN", "ADJ"]
            and not token.is_stop
            and not token.is_punct
            and len(token.lemma_) >= min_length
            and token.lemma_.lower() not in stopwords_set
        ):
            keywords.append(token.lemma_.lower())

    # Count frequencies
    word_freq = FreqDist(keywords)

    # Get top K keywords
    return [word for word, _ in word_freq.most_common(top_k)]


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
        ".pdf": "document",
        ".docx": "document",
        ".pptx": "presentation",
        ".txt": "text",
        ".html": "webpage",
        ".htm": "webpage",
    }

    category = format_map.get(ext, "unknown")

    # Content-based refinement
    if any(kw in text_lower for kw in ["invoice", "receipt", "payment", "bill"]):
        return "invoice"
    elif any(kw in text_lower for kw in ["slide", "presentation", "deck"]):
        return "presentation"
    elif any(kw in text_lower for kw in ["code", "function", "class", "import", "def "]):
        return "code"
    elif any(kw in text_lower for kw in ["policy", "terms", "agreement", "legal"]):
        return "policy"
    elif any(kw in text_lower for kw in ["email", "mail", "@"]):
        return "email"
    elif any(kw in text_lower for kw in ["question", "answer", "qa", "quiz"]):
        return "qa"

    return category


def detect_language(text: str) -> Optional[str]:
    """Detect language using spacy language detection. Falls back to heuristics."""
    nlp = _load_spacy_model()

    if isinstance(nlp, str):
        # Fallback to simple heuristics
        text_lower = text.lower()
        english_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with"]
        spanish_words = ["el", "la", "de", "que", "y", "a", "en", "un", "ser", "se"]
        french_words = ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"]

        english_count = sum(1 for word in english_words if word in text_lower)
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        french_count = sum(1 for word in french_words if word in text_lower)

        if english_count > spanish_count and english_count > french_count:
            return "en"
        elif spanish_count > french_count:
            return "es"
        elif french_count > 0:
            return "fr"
        return "en"

    # Use spacy's language detection
    doc = nlp(text[:10000])  # Sample first 10k chars
    return doc.lang_ if hasattr(doc, "lang_") else "en"


def detect_pii(text: str) -> List[str]:
    """Detect potential PII using spacy NER and regex patterns."""
    warnings = []
    nlp = _load_spacy_model()

    if not isinstance(nlp, str):
        # Use spacy NER for person names, organizations, locations
        doc = nlp(text[:100000])
        entity_types = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                warnings.append("potential_person_name")
                entity_types.add("PERSON")
            elif ent.label_ == "ORG":
                entity_types.add("ORG")
            elif ent.label_ in ["GPE", "LOC"]:
                entity_types.add("LOCATION")

        # Remove duplicates
        if "PERSON" in entity_types:
            warnings = [w for w in warnings if w == "potential_person_name"]
            if warnings:
                warnings = ["potential_person_name"]

    # Enhanced regex patterns
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    if re.search(email_pattern, text):
        warnings.append("potential_email")

    # Phone number patterns (multiple formats)
    phone_patterns = [
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # US format
        r"\+\d{1,3}\s?\(?\d{1,4}\)?[\s.-]?\d{1,4}[\s.-]?\d{1,4}",  # International
    ]
    for pattern in phone_patterns:
        if re.search(pattern, text):
            if "potential_phone" not in warnings:
                warnings.append("potential_phone")
            break

    # SSN pattern (US format)
    ssn_pattern = r"\b\d{3}-?\d{2}-?\d{4}\b"
    if re.search(ssn_pattern, text):
        warnings.append("potential_ssn")

    # Credit card pattern (simplified)
    cc_pattern = r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"
    if re.search(cc_pattern, text):
        warnings.append("potential_cc")

    # IP addresses
    ip_pattern = r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    if re.search(ip_pattern, text):
        warnings.append("potential_ip_address")

    return warnings


def analyze_text_quality(text: str) -> List[str]:
    """Analyze text quality and return warnings"""
    warnings = []

    if len(text) < 50:
        warnings.append("very_short")

    # Check symbol ratio
    if len(text) > 0:
        symbol_ratio = len(re.findall(r"[^\w\s]", text)) / len(text)
        if symbol_ratio > 0.3:
            warnings.append("high_symbol_ratio")

    # Check for mostly whitespace
    if len(text.strip()) < len(text) * 0.5:
        warnings.append("mostly_whitespace")

    return warnings


def compute_readability_score(text: str) -> float:
    """Compute readability score using spacy and NLTK (0-10 scale, higher = more difficult).

    Uses:
    - Spacy for accurate sentence and token parsing
    - NLTK for sentence tokenization
    - Flesch-Kincaid inspired scoring

    Returns:
        Readability score (0-10, higher = more difficult)
    """
    if not text or len(text.strip()) == 0:
        return 5.0

    nlp = _load_spacy_model()
    nltk_error = _ensure_nltk_data()

    # Get sentences
    if nltk_error:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
    else:
        sentences = sent_tokenize(text[:50000])

    if not sentences:
        return 5.0

    # Get words using spacy if available
    if isinstance(nlp, str):
        # Fallback to regex
        words = re.findall(r"\b\w+\b", text.lower())
    else:
        doc = nlp(text[:50000])
        words = [token.text.lower() for token in doc if token.is_alpha]

    if not words:
        return 5.0

    # Compute metrics
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

    # Approximate syllables (simple heuristic: count vowel groups)
    def count_syllables(word: str) -> int:
        word = word.lower()
        if len(word) <= 3:
            return 1
        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        # Adjust for silent e
        if word.endswith("e"):
            count -= 1
        return max(1, count)

    avg_syllables = sum(count_syllables(w) for w in words) / len(words) if words else 0

    # Compute readability score (simplified Flesch-like)
    # Higher values = more difficult
    difficulty = (
        (avg_sentence_length / 20.0) * 3.0  # Sentence length component
        + (avg_word_length / 6.0) * 2.0  # Word length component
        + (avg_syllables / 2.0) * 5.0  # Syllable component
    )

    # Normalize to 0-10 scale
    score = min(10.0, max(0.0, difficulty))
    return round(score, 1)


def detect_tone(text: str) -> Optional[str]:
    """Detect tone using spacy POS tagging and linguistic features.

    Returns:
        Tone label (e.g., "formal", "casual", "technical", "conversational")
    """
    nlp = _load_spacy_model()

    if isinstance(nlp, str):
        # Fallback to simple word matching
        text_lower = text.lower()
        formal_words = ["therefore", "furthermore", "moreover", "consequently", "hence", "thus"]
        casual_words = ["hey", "gonna", "wanna", "yeah", "okay", "cool", "awesome"]
        technical_words = [
            "algorithm",
            "implementation",
            "function",
            "variable",
            "parameter",
            "method",
        ]
        conversational_words = ["you", "your", "we", "our", "let's", "think about"]

        formal_count = sum(1 for word in formal_words if word in text_lower)
        casual_count = sum(1 for word in casual_words if word in text_lower)
        technical_count = sum(1 for word in technical_words if word in text_lower)
        conversational_count = sum(1 for word in conversational_words if word in text_lower)

        if technical_count > 3:
            return "technical"
        elif formal_count > casual_count and formal_count > 1:
            return "formal"
        elif casual_count > 0:
            return "casual"
        elif conversational_count > 5:
            return "conversational"
        return "neutral"

    # Use spacy for advanced analysis
    doc = nlp(text[:50000])
    text_lower = text.lower()

    # Count linguistic features
    formal_conjunctions = ["therefore", "furthermore", "moreover", "consequently", "hence", "thus"]
    casual_contractions = ["gonna", "wanna", "yeah", "okay", "cool", "awesome"]
    technical_terms = ["algorithm", "implementation", "function", "variable", "parameter", "method"]

    formal_count = sum(1 for word in formal_conjunctions if word in text_lower)
    casual_count = sum(1 for word in casual_contractions if word in text_lower)
    technical_count = sum(1 for word in technical_terms if word in text_lower)

    # Analyze POS patterns
    pronouns = [token for token in doc if token.pos_ == "PRON"]
    personal_pronouns = [
        p for p in pronouns if p.text.lower() in ["you", "your", "we", "our", "i", "my"]
    ]

    # Determine tone
    if technical_count > 3:
        return "technical"
    elif formal_count > casual_count and formal_count > 1:
        return "formal"
    elif casual_count > 0:
        return "casual"
    elif len(personal_pronouns) > len(doc) * 0.05:  # More than 5% personal pronouns
        return "conversational"

    return "neutral"


def detect_register(text: str) -> Optional[str]:
    """Detect register using spacy entity recognition and domain-specific vocabulary.

    Returns:
        Register label (e.g., "academic", "business", "informal")
    """
    nlp = _load_spacy_model()
    text_lower = text.lower()

    # Domain-specific vocabulary
    academic_words = [
        "research",
        "study",
        "analysis",
        "hypothesis",
        "methodology",
        "findings",
        "empirical",
        "theoretical",
        "experiment",
        "conclusion",
    ]
    business_words = [
        "revenue",
        "profit",
        "customer",
        "market",
        "strategy",
        "business",
        "stakeholder",
        "roi",
        "investment",
        "growth",
    ]
    technical_words = [
        "software",
        "hardware",
        "system",
        "network",
        "database",
        "api",
        "code",
        "programming",
        "development",
    ]

    academic_count = sum(1 for word in academic_words if word in text_lower)
    business_count = sum(1 for word in business_words if word in text_lower)
    technical_count = sum(1 for word in technical_words if word in text_lower)

    if not isinstance(nlp, str):
        # Use spacy entities for additional context
        doc = nlp(text[:50000])
        org_count = sum(1 for ent in doc.ents if ent.label_ == "ORG")
        money_count = sum(1 for ent in doc.ents if ent.label_ == "MONEY")

        # Business register often has many ORG and MONEY entities
        if money_count > 2 or org_count > 3:
            business_count += 2

    # Determine register
    if academic_count > 2:
        return "academic"
    elif business_count > 2:
        return "business"
    elif technical_count > 2:
        return "technical"

    return "general"
