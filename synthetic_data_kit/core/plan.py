# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Build plan.json from analysis + planning.yaml

from pathlib import Path
from typing import Dict, Any, Optional
import json

from synthetic_data_kit.types.planning_types import (
    PlanSpec,
    MixQuotas,
    CoverageTargets,
    DifficultyTargets,
    StyleTargets,
    SamplerConfig,
    PromptConfig,
    SafetyRules,
    DedupRules,
    QualityGates,
    Budget,
    FilePlan,
)
from synthetic_data_kit.types.analysis_types import AnalysisReport


def build_plan(analysis_path: Path, config: Dict[str, Any]) -> PlanSpec:
    """Build plan.json from analysis report and planning config.

    Args:
        analysis_path: Path to analysis.json report
        config: Configuration dictionary with planning section

    Returns:
        PlanSpec object
    """
    # Load analysis report
    with open(analysis_path, "r", encoding="utf-8") as f:
        analysis_data = json.load(f)

    analysis_report = AnalysisReport(
        root=Path(analysis_data["root"]),
        files=[],  # Will be populated from dict
        aggregate=analysis_data.get("aggregate", {}),
        stats=analysis_data.get("stats", {}),
    )

    # Get planning config
    planning_config = config.get("planning", {})

    # Build plan spec
    plan = PlanSpec()

    # Dataset goal
    plan.dataset_goal = planning_config.get("dataset_goal")

    # Mix quotas
    mix_quotas_dict = planning_config.get("mix_quotas", {})
    plan.mix_quotas = MixQuotas(
        qa=mix_quotas_dict.get("qa", 0.7),
        cot=mix_quotas_dict.get("cot", 0.2),
        summary=mix_quotas_dict.get("summary", 0.1),
    )
    plan.mix_quotas.normalize()

    # Coverage targets
    coverage_targets_dict = planning_config.get("coverage_targets", {})
    plan.coverage_targets = CoverageTargets(
        domains=coverage_targets_dict.get("domains", []),
        topics=coverage_targets_dict.get("topics", []),
    )

    # Difficulty targets
    difficulty_targets_dict = planning_config.get("difficulty_targets", {})
    plan.difficulty_targets = DifficultyTargets(
        min_difficulty=difficulty_targets_dict.get("min_difficulty", 3),
        max_difficulty=difficulty_targets_dict.get("max_difficulty", 8),
        distribution=difficulty_targets_dict.get("distribution", "balanced"),
    )

    # Style targets
    style_targets_dict = planning_config.get("style_targets", {})
    plan.style_targets = StyleTargets(
        tones=style_targets_dict.get("tones", []), registers=style_targets_dict.get("registers", [])
    )

    # Sampler config
    sampler_dict = planning_config.get("sampler", {})
    plan.sampler = SamplerConfig(
        strategy=sampler_dict.get("strategy", "balanced"), weights=sampler_dict.get("weights", {})
    )

    # Prompt config
    prompts_dict = planning_config.get("prompts", {})
    plan.prompts = PromptConfig(
        families=prompts_dict.get("families", []),
        rotation=prompts_dict.get("rotation", "round-robin"),
        per_task_overrides=prompts_dict.get("per_task_overrides", {}),
    )

    # Safety rules
    safety_rules_dict = planning_config.get("safety_rules", {})
    plan.safety_rules = SafetyRules(
        enforce_pii_filter=safety_rules_dict.get("enforce_pii_filter", True),
        enforce_blocklist=safety_rules_dict.get("enforce_blocklist", True),
    )

    # Dedup rules
    dedup_rules_dict = planning_config.get("dedup_rules", {})
    plan.dedup_rules = DedupRules(
        pre_check=dedup_rules_dict.get("pre_check", True),
        post_check=dedup_rules_dict.get("post_check", True),
    )

    # Quality gates
    gates_dict = planning_config.get("gates", {})
    plan.gates = QualityGates(
        min_quality=gates_dict.get("min_quality", 7.0),
        min_coverage=gates_dict.get("min_coverage", 0.5),
    )

    # Budget
    budget_dict = planning_config.get("budget", {})
    plan.budget = Budget(
        max_examples=budget_dict.get("max_examples"), max_tokens=budget_dict.get("max_tokens")
    )

    # Build file plans from analysis
    file_analyses = analysis_data.get("files", [])
    total_files = len(file_analyses)

    # Get budget constraints
    max_examples = plan.budget.max_examples
    if max_examples:
        # Distribute examples across files
        examples_per_file = max(1, max_examples // total_files) if total_files > 0 else 0
    else:
        # Use default from generation config
        generation_config = config.get("generation", {})
        default_num_pairs = generation_config.get("num_pairs", 25)
        examples_per_file = default_num_pairs

    # Prompt rotation
    prompt_families = plan.prompts.families if plan.prompts.families else ["default"]
    rotation_strategy = plan.prompts.rotation

    # Build file plans
    for i, file_data in enumerate(file_analyses):
        file_path = file_data.get("path", "")

        # Compute quotas per file based on mix
        num_qa = int(examples_per_file * plan.mix_quotas.qa)
        num_cot = int(examples_per_file * plan.mix_quotas.cot)
        num_summary = int(examples_per_file * plan.mix_quotas.summary)

        # Determine prompt family (round-robin)
        if rotation_strategy == "round-robin":
            prompt_family = prompt_families[i % len(prompt_families)]
        else:
            prompt_family = prompt_families[0] if prompt_families else None

        # Determine sampler weight
        sampler_weight = 1.0
        if plan.sampler.strategy == "weighted":
            # Use coverage or difficulty to weight
            coverage = file_data.get("coverage")
            if coverage and coverage in plan.sampler.weights:
                sampler_weight = plan.sampler.weights[coverage]
            else:
                sampler_weight = plan.sampler.weights.get(file_path, 1.0)

        # Determine coverage/difficulty/style targets
        coverage_target = file_data.get("coverage")
        difficulty_target = None
        difficulty_score = file_data.get("difficulty")
        if difficulty_score is not None:
            if difficulty_score < plan.difficulty_targets.min_difficulty:
                difficulty_target = "easy"
            elif difficulty_score > plan.difficulty_targets.max_difficulty:
                difficulty_target = "hard"
            else:
                difficulty_target = "medium"

        style_target = file_data.get("tone") or file_data.get("register")

        file_plan = FilePlan(
            file_path=file_path,
            num_qa=num_qa,
            num_cot=num_cot,
            num_summary=num_summary,
            prompt_family=prompt_family,
            sampler_weight=sampler_weight,
            coverage_target=coverage_target,
            difficulty_target=difficulty_target,
            style_target=style_target,
        )

        plan.file_plans.append(file_plan)

    return plan
