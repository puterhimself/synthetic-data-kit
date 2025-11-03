# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Unit tests for enhanced text analysis with spacy and nltk
import pytest
from synthetic_data_kit.utils.text_analysis import (
    initialize_nlp_models,
    extract_keywords,
    detect_pii,
    compute_readability_score,
    detect_tone,
    detect_register,
    estimate_token_count,
)


@pytest.mark.unit
def test_initialize_nlp_models() -> None:
    """Test NLP model initialization"""
    status = initialize_nlp_models()
    
    assert isinstance(status, dict)
    assert 'nltk' in status
    assert 'spacy' in status
    assert 'lemmatizer' in status
    assert 'stopwords' in status
    
    # All values should be None (success) or error strings
    for key, value in status.items():
        assert value is None or isinstance(value, str)


@pytest.mark.unit
def test_enhanced_keyword_extraction() -> None:
    """Test keyword extraction with spacy/nltk"""
    text = """
    Artificial intelligence and machine learning are transforming the technology industry.
    Deep learning models use neural networks to process complex data.
    Natural language processing enables computers to understand human language.
    """
    
    keywords = extract_keywords(text, top_k=10)
    
    assert isinstance(keywords, list)
    assert len(keywords) <= 10
    assert len(keywords) > 0
    
    # Should extract meaningful nouns/adjectives
    keywords_lower = [k.lower() for k in keywords]
    technical_terms = ['intelligence', 'learning', 'network', 'language', 'data', 'model']
    
    # At least some technical terms should be extracted
    found_technical = any(term in keywords_lower for term in technical_terms)
    assert found_technical


@pytest.mark.unit
def test_enhanced_pii_detection() -> None:
    """Test enhanced PII detection with NER"""
    text = """
    John Smith works at Acme Corporation. You can reach him at john.smith@acme.com
    or call 555-123-4567. His office is in New York.
    """
    
    warnings = detect_pii(text)
    
    assert isinstance(warnings, list)
    # Should detect email
    assert 'potential_email' in warnings
    # Should detect phone
    assert 'potential_phone' in warnings
    
    # Test IP address detection
    text_with_ip = "The server is at 192.168.1.1"
    warnings_ip = detect_pii(text_with_ip)
    assert 'potential_ip_address' in warnings_ip


@pytest.mark.unit
def test_readability_score() -> None:
    """Test readability scoring"""
    # Simple text (should be easier)
    simple_text = "The cat sat on the mat. It was a nice day. The sun was bright."
    simple_score = compute_readability_score(simple_text)
    
    # Complex text (should be harder)
    complex_text = """
    The implementation of sophisticated algorithmic methodologies necessitates 
    comprehensive understanding of computational complexity theories and their 
    practical implications within distributed systems architectures.
    """
    complex_score = compute_readability_score(complex_text)
    
    assert isinstance(simple_score, float)
    assert isinstance(complex_score, float)
    assert 0 <= simple_score <= 10
    assert 0 <= complex_score <= 10
    # Complex text should have higher difficulty score
    assert complex_score >= simple_score


@pytest.mark.unit
def test_tone_detection() -> None:
    """Test tone detection with spacy"""
    formal_text = """
    Therefore, we must conclude that the aforementioned hypothesis is valid.
    Furthermore, the empirical evidence supports this assertion.
    """
    formal_tone = detect_tone(formal_text)
    assert formal_tone in ['formal', 'neutral', 'academic']
    
    casual_text = """
    Hey there! This is gonna be awesome. Yeah, it's really cool stuff.
    """
    casual_tone = detect_tone(casual_text)
    assert casual_tone in ['casual', 'conversational']
    
    technical_text = """
    The algorithm uses a hash function to map data to memory locations.
    The implementation requires careful parameter tuning for optimal performance.
    """
    technical_tone = detect_tone(technical_text)
    assert technical_tone in ['technical', 'formal']


@pytest.mark.unit
def test_register_detection() -> None:
    """Test register detection with entity recognition"""
    academic_text = """
    This study examines the hypothesis that research methodology impacts findings.
    Our analysis reveals significant correlations in the experimental data.
    """
    academic_register = detect_register(academic_text)
    assert academic_register in ['academic', 'general']
    
    business_text = """
    Our revenue increased by 25% this quarter. The market strategy yielded 
    significant ROI. Stakeholder investment has grown substantially.
    """
    business_register = detect_register(business_text)
    assert business_register in ['business', 'general']
    
    technical_text = """
    The software system uses a distributed database architecture.
    The API endpoints handle network requests efficiently.
    """
    technical_register = detect_register(technical_text)
    assert technical_register in ['technical', 'general']


@pytest.mark.unit
def test_enhanced_token_counting() -> None:
    """Test token counting with spacy"""
    text = "This is a test sentence with multiple words and punctuation marks!"
    word_count = len(text.split())

    token_count = estimate_token_count(text)

    assert isinstance(token_count, int)
    assert token_count > 0
    # Should be roughly the number of whitespace-delimited words plus punctuation tokens
    assert word_count <= token_count <= word_count + 6


@pytest.mark.unit
def test_fallback_mechanisms() -> None:
    """Test that functions work even if spacy/nltk fail"""
    # These should not raise exceptions even if models aren't loaded
    text = "Sample text for fallback testing"
    
    keywords = extract_keywords(text)
    assert isinstance(keywords, list)
    
    warnings = detect_pii(text)
    assert isinstance(warnings, list)
    
    score = compute_readability_score(text)
    assert isinstance(score, float)
    
    tone = detect_tone(text)
    assert tone is None or isinstance(tone, str)
    
    register = detect_register(text)
    assert register is None or isinstance(register, str)


@pytest.mark.unit
def test_large_text_handling() -> None:
    """Test that large texts are handled efficiently"""
    # Create a large text (over 100k chars)
    large_text = "This is a sample sentence. " * 5000  # ~135k chars
    
    # These should complete without errors or excessive memory usage
    keywords = extract_keywords(large_text, top_k=15)
    assert isinstance(keywords, list)
    assert len(keywords) <= 15
    
    score = compute_readability_score(large_text)
    assert isinstance(score, float)
    
    tone = detect_tone(large_text)
    assert tone is None or isinstance(tone, str)


@pytest.mark.unit
def test_empty_text_handling() -> None:
    """Test handling of empty or minimal text"""
    empty_text = ""
    
    keywords = extract_keywords(empty_text)
    assert isinstance(keywords, list)
    assert len(keywords) == 0
    
    score = compute_readability_score(empty_text)
    assert isinstance(score, float)
    assert score == 5.0  # Default score
    
    warnings = detect_pii(empty_text)
    assert isinstance(warnings, list)
    assert len(warnings) == 0


