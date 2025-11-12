# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Planning types and data structures

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import json


@dataclass
class MixQuotas:
    """Mix quotas for different content types"""

    qa: float = 0.7
    cot: float = 0.2
    summary: float = 0.1

    def normalize(self):
        """Normalize quotas to sum to 1.0"""
        total = self.qa + self.cot + self.summary
        if total > 0:
            self.qa /= total
            self.cot /= total
            self.summary /= total


@dataclass
class CoverageTargets:
    """Coverage targets for planning"""

    domains: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


@dataclass
class DifficultyTargets:
    """Difficulty targets for planning"""

    min_difficulty: int = 3
    max_difficulty: int = 8
    distribution: str = "balanced"  # "easy", "balanced", "hard"


@dataclass
class StyleTargets:
    """Style targets for planning"""

    tones: List[str] = field(default_factory=list)
    registers: List[str] = field(default_factory=list)


@dataclass
class SamplerConfig:
    """Sampler configuration"""

    strategy: str = "balanced"  # "balanced", "weighted", "random"
    weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class PromptConfig:
    """Prompt configuration"""

    families: List[str] = field(default_factory=list)
    rotation: str = "round-robin"  # "round-robin", "random", "weighted"
    per_task_overrides: Dict[str, str] = field(default_factory=dict)


@dataclass
class SafetyRules:
    """Safety rules configuration"""

    enforce_pii_filter: bool = True
    enforce_blocklist: bool = True


@dataclass
class DedupRules:
    """Deduplication rules configuration"""

    pre_check: bool = True
    post_check: bool = True


@dataclass
class QualityGates:
    """Quality gates configuration"""

    min_quality: float = 7.0
    min_coverage: float = 0.5


@dataclass
class Budget:
    """Budget constraints"""

    max_examples: Optional[int] = None
    max_tokens: Optional[int] = None


@dataclass
class FilePlan:
    """Plan for a single file"""

    file_path: str
    num_qa: int = 0
    num_cot: int = 0
    num_summary: int = 0
    prompt_family: Optional[str] = None
    sampler_weight: float = 1.0
    coverage_target: Optional[str] = None
    difficulty_target: Optional[str] = None
    style_target: Optional[str] = None


@dataclass
class PlanSpec:
    """Complete plan specification"""

    dataset_goal: Optional[str] = None
    mix_quotas: MixQuotas = field(default_factory=MixQuotas)
    coverage_targets: CoverageTargets = field(default_factory=CoverageTargets)
    difficulty_targets: DifficultyTargets = field(default_factory=DifficultyTargets)
    style_targets: StyleTargets = field(default_factory=StyleTargets)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    safety_rules: SafetyRules = field(default_factory=SafetyRules)
    dedup_rules: DedupRules = field(default_factory=DedupRules)
    gates: QualityGates = field(default_factory=QualityGates)
    budget: Budget = field(default_factory=Budget)
    file_plans: List[FilePlan] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {}
        result["dataset_goal"] = self.dataset_goal
        result["mix_quotas"] = asdict(self.mix_quotas)
        result["coverage_targets"] = asdict(self.coverage_targets)
        result["difficulty_targets"] = asdict(self.difficulty_targets)
        result["style_targets"] = asdict(self.style_targets)
        result["sampler"] = asdict(self.sampler)
        result["prompts"] = asdict(self.prompts)
        result["safety_rules"] = asdict(self.safety_rules)
        result["dedup_rules"] = asdict(self.dedup_rules)
        result["gates"] = asdict(self.gates)
        result["budget"] = asdict(self.budget)
        result["file_plans"] = [asdict(fp) for fp in self.file_plans]
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, output_path: Path) -> None:
        """Save plan to JSON file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanSpec":
        """Create PlanSpec from dictionary"""
        plan = cls()
        plan.dataset_goal = data.get("dataset_goal")

        if "mix_quotas" in data:
            plan.mix_quotas = MixQuotas(**data["mix_quotas"])

        if "coverage_targets" in data:
            plan.coverage_targets = CoverageTargets(**data["coverage_targets"])

        if "difficulty_targets" in data:
            plan.difficulty_targets = DifficultyTargets(**data["difficulty_targets"])

        if "style_targets" in data:
            plan.style_targets = StyleTargets(**data["style_targets"])

        if "sampler" in data:
            plan.sampler = SamplerConfig(**data["sampler"])

        if "prompts" in data:
            plan.prompts = PromptConfig(**data["prompts"])

        if "safety_rules" in data:
            plan.safety_rules = SafetyRules(**data["safety_rules"])

        if "dedup_rules" in data:
            plan.dedup_rules = DedupRules(**data["dedup_rules"])

        if "gates" in data:
            plan.gates = QualityGates(**data["gates"])

        if "budget" in data:
            plan.budget = Budget(**data["budget"])

        if "file_plans" in data:
            plan.file_plans = [FilePlan(**fp) for fp in data["file_plans"]]

        return plan

    @classmethod
    def load(cls, input_path: Path) -> "PlanSpec":
        """Load plan from JSON file"""
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
