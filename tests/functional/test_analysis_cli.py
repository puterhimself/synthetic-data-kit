# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Functional tests for analysis CLI command
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from typer.testing import CliRunner

from synthetic_data_kit.cli import app


@pytest.mark.functional
def test_analyze_command_single_file(patch_config, tmp_path):
    """Test analyze command with a single file"""
    runner = CliRunner()
    
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document with some content about Python programming and data science.")
    
    # Create temporary output file
    output_file = tmp_path / "analysis.json"
    
    result = runner.invoke(app, [
        "analyze",
        str(test_file),
        "--output",
        str(output_file)
    ])
    
    assert result.exit_code == 0
    assert output_file.exists()
    
    # Check that output is valid JSON
    with open(output_file) as f:
        report = json.load(f)
    
    assert "root" in report
    assert "files" in report
    assert "aggregate" in report
    assert "stats" in report
    assert len(report["files"]) == 1
    assert report["files"][0]["path"] == str(test_file)


@pytest.mark.functional
def test_analyze_command_directory(patch_config, tmp_path):
    """Test analyze command with a directory"""
    runner = CliRunner()
    
    # Create test directory with files
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()
    
    (test_dir / "doc1.txt").write_text("First document about machine learning.")
    (test_dir / "doc2.txt").write_text("Second document about data analysis.")
    
    output_file = tmp_path / "analysis.json"
    
    result = runner.invoke(app, [
        "analyze",
        str(test_dir),
        "--output",
        str(output_file),
        "--max-files",
        "10"
    ])
    
    assert result.exit_code == 0
    assert output_file.exists()
    
    with open(output_file) as f:
        report = json.load(f)
    
    assert len(report["files"]) == 2
    assert report["stats"]["num_files_analyzed"] == 2


@pytest.mark.functional
def test_analyze_command_skip_analysis(patch_config, tmp_path):
    """Test analyze command with --skip-analysis flag"""
    runner = CliRunner()
    
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    result = runner.invoke(app, [
        "analyze",
        str(test_file),
        "--skip-analysis"
    ])
    
    # Should exit early with success
    assert result.exit_code == 0
    assert "Skipping analysis" in result.stdout


@pytest.mark.functional
def test_analyze_command_config_integration(patch_config, tmp_path):
    """Test that analyze command respects config settings"""
    runner = CliRunner()
    
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content for analysis")
    
    output_file = tmp_path / "analysis.json"
    
    # Test with --no-cache flag
    result = runner.invoke(app, [
        "analyze",
        str(test_file),
        "--output",
        str(output_file),
        "--no-cache"
    ])
    
    assert result.exit_code == 0
    assert output_file.exists()


@pytest.mark.functional
def test_analyze_command_output_stdout(patch_config, tmp_path):
    """Test analyze command outputs to stdout when no output file specified"""
    runner = CliRunner()
    
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    
    result = runner.invoke(app, [
        "analyze",
        str(test_file)
    ])
    
    assert result.exit_code == 0
    # Should contain JSON output
    assert "root" in result.stdout or "files" in result.stdout

