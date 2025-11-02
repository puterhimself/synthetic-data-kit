# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import json
import tempfile
from pathlib import Path
from synthetic_data_kit.core.analysis import analyze_path, AnalysisOptions
from synthetic_data_kit.core.plan import build_plan
from synthetic_data_kit.types.planning_types import PlanSpec


@pytest.mark.integration
def test_analyze_plan_create_workflow(tmp_path):
    """Integration test: ingest→analyze→plan→create --plan→curate"""
    # This is a simplified integration test that verifies the workflow
    # In a real scenario, you would run the actual commands
    
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("""
    Machine learning is a subset of artificial intelligence.
    It involves algorithms that can learn from data.
    Neural networks are a popular type of machine learning model.
    """)
    
    # Step 1: Analyze
    options = AnalysisOptions(
        use_llm=False,
        max_chars=10000,
        output_path=tmp_path / "analysis.json",
        persist_cache=True
    )
    
    config = {
        "analysis": {
            "coverage": {"enabled": True},
            "difficulty": {"enabled": True},
            "style": {"enabled": True},
            "dedup": {"enabled": True}
        }
    }
    
    report = analyze_path(test_file, options, config)
    assert report is not None
    assert len(report.files) > 0
    
    # Step 2: Plan
    analysis_path = tmp_path / "analysis.json"
    planning_config = {
        "planning": {
            "dataset_goal": "Test dataset",
            "mix_quotas": {"qa": 0.7, "cot": 0.2, "summary": 0.1},
            "gates": {"min_quality": 7.0}
        },
        "generation": {"num_pairs": 25}
    }
    
    plan = build_plan(analysis_path, planning_config)
    assert plan is not None
    assert len(plan.file_plans) > 0
    
    # Step 3: Verify plan can be saved/loaded
    plan_path = tmp_path / "plan.json"
    plan.save(plan_path)
    
    plan2 = PlanSpec.load(plan_path)
    assert plan2.dataset_goal == plan.dataset_goal
    assert len(plan2.file_plans) == len(plan.file_plans)
    
    # Step 4: Verify plan has correct structure
    assert plan.mix_quotas.qa > 0
    assert plan.gates.min_quality == 7.0
    
    # Note: Actual create and curate steps would require LLM setup
    # This test verifies the workflow up to plan creation

