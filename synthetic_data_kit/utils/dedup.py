# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Deduplication: MinHash LSH first, embedding tie-break at cosine >= 0.92

from typing import List, Set, Tuple, Optional, Dict, Any
import hashlib
from dataclasses import dataclass
import re


@dataclass
class MinHashSignature:
    """MinHash signature for a document"""
    hash_values: List[int]
    num_perm: int = 128  # Number of permutations for MinHash


def compute_minhash(text: str, num_perm: int = 128) -> MinHashSignature:
    """Compute MinHash signature for text using shingles.
    
    Args:
        text: Text to hash
        num_perm: Number of permutations
        
    Returns:
        MinHashSignature object
    """
    # Create shingles (3-grams of words)
    words = re.findall(r'\b\w+\b', text.lower())
    shingles = set()
    
    for i in range(len(words) - 2):
        shingle = ' '.join(words[i:i+3])
        shingles.add(shingle)
    
    # Compute hash values for each shingle
    hash_values = []
    for i in range(num_perm):
        min_hash = min(
            int(hashlib.md5(f"{shingle}_{i}".encode()).hexdigest(), 16)
            for shingle in shingles
        ) if shingles else 0
        hash_values.append(min_hash)
    
    return MinHashSignature(hash_values, num_perm)


def jaccard_similarity_from_minhash(sig1: MinHashSignature, sig2: MinHashSignature) -> float:
    """Compute Jaccard similarity from MinHash signatures.
    
    Args:
        sig1: First MinHash signature
        sig2: Second MinHash signature
        
    Returns:
        Jaccard similarity estimate (0-1)
    """
    if sig1.num_perm != sig2.num_perm:
        raise ValueError("Signatures must have same num_perm")
    
    matches = sum(1 for h1, h2 in zip(sig1.hash_values, sig2.hash_values) if h1 == h2)
    return matches / sig1.num_perm


class MinHashLSH:
    """Locality Sensitive Hashing for MinHash signatures"""
    
    def __init__(self, threshold: float = 0.5, num_bands: int = 20):
        """Initialize LSH index.
        
        Args:
            threshold: Similarity threshold for candidate pairs
            num_bands: Number of bands for LSH
        """
        self.threshold = threshold
        self.num_bands = num_bands
        self.bands: List[Dict[str, Set[int]]] = [{} for _ in range(num_bands)]
        self.documents: Dict[int, str] = {}
        self.signatures: Dict[int, MinHashSignature] = {}
        self.next_id = 0
    
    def add(self, text: str, doc_id: Optional[int] = None) -> int:
        """Add a document to the LSH index.
        
        Args:
            text: Document text
            doc_id: Optional document ID (auto-assigned if None)
            
        Returns:
            Document ID
        """
        if doc_id is None:
            doc_id = self.next_id
            self.next_id += 1
        
        self.documents[doc_id] = text
        sig = compute_minhash(text)
        self.signatures[doc_id] = sig
        
        # Add to bands
        rows_per_band = len(sig.hash_values) // self.num_bands
        for band_idx in range(self.num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_hash = hash(tuple(sig.hash_values[start:end]))
            band_key = str(band_hash)
            
            if band_key not in self.bands[band_idx]:
                self.bands[band_idx][band_key] = set()
            self.bands[band_idx][band_key].add(doc_id)
        
        return doc_id
    
    def find_candidates(self, text: str) -> List[Tuple[int, float]]:
        """Find candidate duplicate documents.
        
        Args:
            text: Query text
            
        Returns:
            List of (doc_id, similarity) tuples
        """
        query_sig = compute_minhash(text)
        candidates: Set[int] = set()
        
        # Find candidates in LSH bands
        rows_per_band = len(query_sig.hash_values) // self.num_bands
        for band_idx in range(self.num_bands):
            start = band_idx * rows_per_band
            end = start + rows_per_band
            band_hash = hash(tuple(query_sig.hash_values[start:end]))
            band_key = str(band_hash)
            
            if band_key in self.bands[band_idx]:
                candidates.update(self.bands[band_idx][band_key])
        
        # Compute actual similarities for candidates
        results = []
        for doc_id in candidates:
            sim = jaccard_similarity_from_minhash(query_sig, self.signatures[doc_id])
            if sim >= self.threshold:
                results.append((doc_id, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def cosine_similarity_embedding(embed1: List[float], embed2: List[float]) -> float:
    """Compute cosine similarity between embeddings.
    
    Args:
        embed1: First embedding vector
        embed2: Second embedding vector
        
    Returns:
        Cosine similarity (0-1)
    """
    if len(embed1) != len(embed2):
        raise ValueError("Embeddings must have same length")
    
    dot_product = sum(a * b for a, b in zip(embed1, embed2))
    norm1 = sum(a * a for a in embed1) ** 0.5
    norm2 = sum(b * b for b in embed2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def find_duplicates_minhash_lsh(
    texts: List[str],
    threshold: float = 0.5,
    num_bands: int = 20
) -> List[Tuple[int, int, float]]:
    """Find duplicates using MinHash LSH.
    
    Args:
        texts: List of texts to check
        threshold: Similarity threshold
        num_bands: Number of LSH bands
        
    Returns:
        List of (doc_id1, doc_id2, similarity) tuples
    """
    lsh = MinHashLSH(threshold=threshold, num_bands=num_bands)
    
    # Add all documents
    doc_ids = []
    for text in texts:
        doc_id = lsh.add(text)
        doc_ids.append(doc_id)
    
    # Find duplicates
    duplicates = []
    seen_pairs: Set[Tuple[int, int]] = set()
    
    for i, text in enumerate(texts):
        candidates = lsh.find_candidates(text)
        for candidate_id, sim in candidates:
            if candidate_id != doc_ids[i]:
                pair = tuple(sorted([doc_ids[i], candidate_id]))
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    duplicates.append((doc_ids[i], candidate_id, sim))
    
    return duplicates


def find_duplicates_with_embedding_tiebreak(
    texts: List[str],
    embeddings: Optional[List[List[float]]] = None,
    minhash_threshold: float = 0.5,
    embedding_threshold: float = 0.92,
    num_bands: int = 20
) -> List[Tuple[int, int, float]]:
    """Find duplicates using MinHash LSH, with embedding tie-break.
    
    Args:
        texts: List of texts to check
        embeddings: Optional embeddings for tie-breaking
        minhash_threshold: MinHash similarity threshold
        embedding_threshold: Embedding cosine similarity threshold for tie-break
        num_bands: Number of LSH bands
        
    Returns:
        List of (doc_id1, doc_id2, similarity) tuples
    """
    # First pass: MinHash LSH
    minhash_duplicates = find_duplicates_minhash_lsh(
        texts, threshold=minhash_threshold, num_bands=num_bands
    )
    
    if not embeddings or embedding_threshold <= minhash_threshold:
        return minhash_duplicates
    
    # Second pass: Embedding tie-break for high-similarity pairs
    final_duplicates = []
    for doc_id1, doc_id2, minhash_sim in minhash_duplicates:
        if minhash_sim >= embedding_threshold:
            # Use embedding similarity if available
            if embeddings and doc_id1 < len(embeddings) and doc_id2 < len(embeddings):
                embed_sim = cosine_similarity_embedding(
                    embeddings[doc_id1], embeddings[doc_id2]
                )
                if embed_sim >= embedding_threshold:
                    final_duplicates.append((doc_id1, doc_id2, embed_sim))
            else:
                # No embeddings, use MinHash similarity
                final_duplicates.append((doc_id1, doc_id2, minhash_sim))
        else:
            # Below embedding threshold, keep MinHash result
            final_duplicates.append((doc_id1, doc_id2, minhash_sim))
    
    return final_duplicates

