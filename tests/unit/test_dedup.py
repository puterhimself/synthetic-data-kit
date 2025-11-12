# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from synthetic_data_kit.utils.dedup import (
    compute_minhash,
    MinHashSignature,
    jaccard_similarity_from_minhash,
    MinHashLSH,
    cosine_similarity_embedding,
    find_duplicates_minhash_lsh
)


def test_compute_minhash():
    """Test MinHash computation"""
    text = "This is a test document with some content."
    sig = compute_minhash(text, num_perm=64)
    
    assert isinstance(sig, MinHashSignature)
    assert len(sig.hash_values) == 64
    assert all(isinstance(h, int) for h in sig.hash_values)


def test_minhash_similarity():
    """Test MinHash similarity computation"""
    text1 = "This is a test document."
    text2 = "This is a test document."
    text3 = "Completely different content here."
    
    sig1 = compute_minhash(text1)
    sig2 = compute_minhash(text2)
    sig3 = compute_minhash(text3)
    
    # Identical texts should have high similarity
    sim12 = jaccard_similarity_from_minhash(sig1, sig2)
    assert sim12 > 0.8
    
    # Different texts should have low similarity
    sim13 = jaccard_similarity_from_minhash(sig1, sig3)
    assert sim13 < 0.5


def test_minhash_lsh():
    """Test MinHash LSH indexing"""
    lsh = MinHashLSH(threshold=0.5, num_bands=10)
    
    text1 = "Machine learning is a subset of artificial intelligence."
    text2 = "Machine learning is a subset of artificial intelligence."  # Duplicate
    text3 = "Completely different content here."
    
    doc_id1 = lsh.add(text1)
    doc_id2 = lsh.add(text2)
    doc_id3 = lsh.add(text3)
    
    # Find candidates for text1
    candidates = lsh.find_candidates(text1)
    assert len(candidates) > 0
    
    # Should find duplicate (text2)
    candidate_ids = [c[0] for c in candidates]
    assert doc_id2 in candidate_ids


def test_cosine_similarity_embedding():
    """Test cosine similarity for embeddings"""
    embed1 = [1.0, 0.0, 0.0]
    embed2 = [1.0, 0.0, 0.0]  # Identical
    embed3 = [0.0, 1.0, 0.0]  # Orthogonal
    
    sim = cosine_similarity_embedding(embed1, embed2)
    assert abs(sim - 1.0) < 0.001
    
    sim = cosine_similarity_embedding(embed1, embed3)
    assert abs(sim - 0.0) < 0.001


def test_find_duplicates_minhash_lsh():
    """Test duplicate finding with MinHash LSH"""
    texts = [
        "Machine learning algorithms",
        "Machine learning algorithms",  # Duplicate
        "Completely different text",
        "Machine learning algorithms"  # Another duplicate
    ]
    
    duplicates = find_duplicates_minhash_lsh(texts, threshold=0.5)
    
    # Should find some duplicates
    assert len(duplicates) > 0
    
    # All duplicates should have similarity >= threshold
    for doc_id1, doc_id2, sim in duplicates:
        assert sim >= 0.5

