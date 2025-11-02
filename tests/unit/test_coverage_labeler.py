# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from synthetic_data_kit.utils.coverage_labeler import (
    label_coverage_keyword_rules,
    label_coverage_llm_fallback,
    label_coverage
)


def test_label_coverage_keyword_rules():
    """Test keyword-based coverage labeling"""
    text = "Machine learning algorithms and neural networks are used in AI."
    keywords = ["machine", "learning", "algorithms", "neural", "networks", "AI"]
    
    keyword_rules = {
        "technical": ["algorithm", "neural", "network", "machine learning"],
        "general": ["overview", "introduction", "basics"]
    }
    
    label = label_coverage_keyword_rules(text, keywords, keyword_rules)
    assert label == "technical"
    
    # Test with no match
    label = label_coverage_keyword_rules(text, keywords, {})
    assert label is None


def test_label_coverage_no_rules():
    """Test coverage labeling with no rules"""
    text = "Some text"
    keywords = ["some", "text"]
    
    label = label_coverage_keyword_rules(text, keywords, None)
    assert label is None


def test_label_coverage_hybrid():
    """Test hybrid coverage labeling (keyword first, LLM fallback)"""
    text = "Machine learning algorithms and neural networks."
    keywords = ["machine", "learning", "algorithms"]
    
    keyword_rules = {
        "technical": ["algorithm", "neural"]
    }
    
    # Should use keyword rules
    label = label_coverage(text, keywords, keyword_rules, llm_fallback=False)
    assert label == "technical"
    
    # Without LLM client, should still work with keyword rules
    label = label_coverage(text, keywords, keyword_rules, llm_fallback=True, llm_client=None)
    assert label == "technical"

