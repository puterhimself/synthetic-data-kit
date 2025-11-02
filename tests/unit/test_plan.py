# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import json
import tempfile
from pathlib import Path
from synthetic_data_kit.types.planning_types import (
    PlanSpec, MixQuotas, CoverageTargets, DifficultyTargets, StyleTargets,
    SamplerConfig, PromptConfig, SafetyRules, DedupRules, QualityGates,
    Budget, FilePlan
)
from synthetic_data_kit.core.plan import build_plan


def test_mix_quotas_normalize():
    """Test MixQuotas normalization"""
    quotas = MixQuotas(qa=0.7, cot=0.2, summary=0.1)
    quotas.normalize()
    
    assert abs(quotas.qa + quotas.cot + quotas.summary - 1.0) < 0.001
    
    # Test with different values
    quotas2 = MixQuotas(qa=1.0, cot=1.0, summary=1.0)
    quotas2.normalize()
    assert abs(quotas2.qa + quotas2.cot + quotas2.summary - 1.0) < 0.001


def test_plan_spec_serialization():
    """Test PlanSpec serialization to/from dict"""
    plan = PlanSpec()
    plan.dataset_goal = "Test dataset"
    plan.mix_quotas = MixQuotas(qa=0.8, cot=0.2, summary=0.0)
    plan.file_plans = [
        FilePlan(file_path="test.txt", num_qa=10, num_cot=2)
    ]
    
    # Serialize
    plan_dict = plan.to_dict()
    assert plan_dict['dataset_goal'] == "Test dataset"
    assert plan_dict['mix_quotas']['qa'] == 0.8
    assert len(plan_dict['file_plans']) == 1
    
    # Deserialize
    plan2 = PlanSpec.from_dict(plan_dict)
    assert plan2.dataset_goal == "Test dataset"
    assert plan2.mix_quotas.qa == 0.8
    assert len(plan2.file_plans) == 1


def test_plan_spec_save_load():
    """Test PlanSpec save/load to file"""
    plan = PlanSpec()
    plan.dataset_goal = "Test dataset"
    plan.mix_quotas = MixQuotas(qa=0.7, cot=0.3, summary=0.0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = Path(tmpdir) / "plan.json"
        plan.save(plan_path)
        
        assert plan_path.exists()
        
        # Load it back
        plan2 = PlanSpec.load(plan_path)
        assert plan2.dataset_goal == "Test dataset"
        assert plan2.mix_quotas.qa == 0.7


def test_build_plan():
    """Test plan building from analysis"""
    # Create a mock analysis report
    analysis_data = {
        "root": "/test/root",
        "files": [
            {
                "path": "/test/file1.txt",
                "coverage": "technical",
                "difficulty": 5.0,
                "tone": "formal",
                "char_count": 1000
            },
            {
                "path": "/test/file2.txt",
                "coverage": "general",
                "difficulty": 3.0,
                "tone": "casual",
                "char_count": 2000
            }
        ],
        "aggregate": {},
        "stats": {}
    }
    
    config = {
        "planning": {
            "dataset_goal": "Test dataset",
            "mix_quotas": {"qa": 0.7, "cot": 0.2, "summary": 0.1},
            "gates": {"min_quality": 7.0}
        },
        "generation": {"num_pairs": 25}
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        analysis_path = Path(tmpdir) / "analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_data, f)
        
        plan = build_plan(analysis_path, config)
        
        assert plan.dataset_goal == "Test dataset"
        assert len(plan.file_plans) == 2
        assert plan.file_plans[0].file_path == "/test/file1.txt"
        assert plan.file_plans[0].coverage_target == "technical"

