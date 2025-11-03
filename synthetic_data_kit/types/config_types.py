# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Configuration type definitions for analysis
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AnalysisConfig:
    """Configuration for analysis stage"""

    enabled: bool = True
    use_llm: bool = False
    max_chars: int = 100_000
    keyword_top_k: int = 15
    categorize: bool = True
    detect_language: bool = True
    detect_pii: bool = False
    cache: bool = True
    default_output: str = ".sdkit/analysis/report.json"
    max_files: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "AnalysisConfig":
        """Create AnalysisConfig from dictionary"""
        return cls(
            enabled=config_dict.get("enabled", True),
            use_llm=config_dict.get("use_llm", False),
            max_chars=config_dict.get("max_chars", 100_000),
            keyword_top_k=config_dict.get("keyword_top_k", 15),
            categorize=config_dict.get("categorize", True),
            detect_language=config_dict.get("detect_language", True),
            detect_pii=config_dict.get("detect_pii", False),
            cache=config_dict.get("cache", True),
            default_output=config_dict.get("default_output", ".sdkit/analysis/report.json"),
            max_files=config_dict.get("max_files", None),
        )
