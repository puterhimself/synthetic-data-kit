# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Unit tests for text analysis utilities
import pytest
from pathlib import Path
from synthetic_data_kit.utils.text_analysis import (
    normalize_text,
    extract_text_from_parser_result,
    estimate_token_count,
    generate_summary,
    extract_keywords,
    extract_tags,
    categorize_file,
    detect_language,
    detect_pii,
    analyze_text_quality
)


def test_normalize_text():
    """Test text normalization"""
    text = "This   is   a   test   with   extra   spaces."
    normalized = normalize_text(text)
    assert "  " not in normalized  # No double spaces
    
    text2 = "Text with special@#$% characters!"
    normalized2 = normalize_text(text2)
    assert isinstance(normalized2, str)


def test_extract_text_from_parser_result():
    """Test text extraction from various parser result formats"""
    # String format
    result1 = "Simple text string"
    assert extract_text_from_parser_result(result1) == "Simple text string"
    
    # List of dicts format
    result2 = [{"text": "First paragraph"}, {"text": "Second paragraph"}]
    extracted = extract_text_from_parser_result(result2)
    assert "First paragraph" in extracted
    assert "Second paragraph" in extracted
    
    # List of strings
    result3 = ["First", "Second"]
    extracted3 = extract_text_from_parser_result(result3)
    assert "First" in extracted3
    
    # Single dict
    result4 = {"text": "Single text"}
    assert extract_text_from_parser_result(result4) == "Single text"


def test_estimate_token_count():
    """Test token count estimation"""
    text = "This is a test sentence with multiple words."
    count = estimate_token_count(text)
    assert count > 0
    assert isinstance(count, int)


def test_generate_summary():
    """Test summary generation"""
    text = """
    This is the first sentence. It contains important information.
    This is the second sentence. It also has useful content.
    This is the third sentence. It completes the thought.
    This is the fourth sentence. It adds more details.
    """
    summary = generate_summary(text, max_sentences=2)
    assert len(summary) > 0
    assert isinstance(summary, str)


def test_extract_keywords():
    """Test keyword extraction"""
    text = """
    Python is a programming language. Python is used for data science.
    Machine learning uses Python. Data analysis requires Python.
    """
    keywords = extract_keywords(text, top_k=5)
    assert len(keywords) <= 5
    assert "python" in keywords or "Python" in keywords.lower()
    assert all(isinstance(kw, str) for kw in keywords)


def test_extract_tags():
    """Test tag extraction"""
    text = "Python programming language data science machine learning"
    keywords = extract_keywords(text, top_k=10)
    tags = extract_tags(text, keywords)
    assert len(tags) <= 10
    assert all(isinstance(tag, str) for tag in tags)


def test_categorize_file():
    """Test file categorization"""
    file_path = Path("document.pdf")
    text = "This is an invoice for payment of services rendered."
    keywords = ["invoice", "payment"]
    
    category = categorize_file(file_path, text, keywords)
    assert category in ["invoice", "document", "unknown"]
    
    # Test code categorization
    code_text = "def function(): import os class MyClass:"
    code_keywords = ["def", "import", "class"]
    code_category = categorize_file(Path("script.py"), code_text, code_keywords)
    assert code_category == "code"


def test_detect_language():
    """Test language detection"""
    english_text = "This is an English text with common words like the and and."
    language = detect_language(english_text)
    assert language == "en"
    
    # Test with minimal text
    short_text = "Hello"
    lang = detect_language(short_text)
    assert lang in ["en", "es", "fr"] or lang is None


def test_detect_pii():
    """Test PII detection"""
    text_with_email = "Contact us at user@example.com for more information."
    warnings = detect_pii(text_with_email)
    assert "potential_email" in warnings
    
    text_with_phone = "Call us at 555-123-4567"
    warnings2 = detect_pii(text_with_phone)
    assert "potential_phone" in warnings2
    
    # Text without PII
    clean_text = "This is a normal text without any sensitive information."
    warnings3 = detect_pii(clean_text)
    assert isinstance(warnings3, list)


def test_analyze_text_quality():
    """Test text quality analysis"""
    # Very short text
    short_text = "Hi"
    warnings = analyze_text_quality(short_text)
    assert "very_short" in warnings
    
    # Normal text
    normal_text = "This is a normal length text with sufficient content to pass quality checks."
    warnings2 = analyze_text_quality(normal_text)
    assert isinstance(warnings2, list)
    
    # Text with high symbol ratio
    symbol_text = "@#$%^&*()" * 10
    warnings3 = analyze_text_quality(symbol_text)
    assert isinstance(warnings3, list)

