# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Analysis types and data structures
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


@dataclass
class AnalysisOptions:
    """Options for controlling analysis behavior"""
    use_llm: bool = False
    max_chars: int = 100_000
    keyword_top_k: int = 15
    categorize: bool = True
    detect_language: bool = True
    detect_pii: bool = False
    cache_enabled: bool = True
    output_path: Optional[Path] = None
    persist_cache: bool = True
    max_files: Optional[int] = None  # Limit number of files to analyze


@dataclass
class FileAnalysis:
    """Analysis results for a single file"""
    path: Path
    format: str
    language: Optional[str] = None
    char_count: int = 0
    token_count: int = 0
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    extraction_time_ms: float = 0.0
    analysis_time_ms: float = 0.0
    truncated: bool = False
    llm_used: bool = False
    
    # Extended analysis fields
    coverage: Optional[str] = None  # Coverage label (domain/topic)
    difficulty: Optional[float] = None  # Readability/difficulty score (0-10)
    safety_warnings: List[str] = field(default_factory=list)  # PII, policy violations
    tone: Optional[str] = None  # Tone (formal, casual, technical, etc.)
    register: Optional[str] = None  # Register (academic, business, etc.)
    duplicate_signature: Optional[str] = None  # MinHash signature for dedup
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['path'] = str(self.path)
        return result


@dataclass
class AnalysisReport:
    """Complete analysis report for a path (file or directory)"""
    root: Path
    files: List[FileAnalysis]
    aggregate: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'root': str(self.root),
            'files': [f.to_dict() for f in self.files],
            'aggregate': self.aggregate,
            'stats': self.stats
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, output_path: Path) -> None:
        """Save report to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

