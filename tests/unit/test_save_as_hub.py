import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synthetic_data_kit.core.save_as import push_to_huggingface_hub


@pytest.fixture()
def sample_jsonl(tmp_path: Path) -> Path:
    file_path = tmp_path / "conversations.jsonl"
    records = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]
    file_path.write_text("\n".join(json.dumps(record) for record in records))
    return file_path


@patch("synthetic_data_kit.core.save_as.Dataset")
@patch("synthetic_data_kit.core.save_as.HfApi")
def test_push_to_hub_uses_dataset_split(mock_hf_api, mock_dataset, sample_jsonl):
    dataset_instance = MagicMock()
    mock_dataset.from_json.return_value = dataset_instance

    result_url = push_to_huggingface_hub(
        str(sample_jsonl),
        repo_id="tester/dataset",
        token="token_123",
    )

    mock_hf_api.assert_called_once()
    mock_dataset.from_json.assert_called_once_with(str(sample_jsonl))
    dataset_instance.push_to_hub.assert_called_once_with(
        repo_id="tester/dataset",
        token="token_123",
        private=False,
        commit_message=None,
        split="conversations",
    )
    assert result_url == "https://huggingface.co/datasets/tester/dataset"


@patch("synthetic_data_kit.core.save_as.Dataset")
@patch("synthetic_data_kit.core.save_as.HfApi")
def test_push_to_hub_sanitizes_split_name(mock_hf_api, mock_dataset, sample_jsonl):
    dataset_instance = MagicMock()
    mock_dataset.from_json.return_value = dataset_instance

    push_to_huggingface_hub(
        str(sample_jsonl),
        repo_id="tester/dataset",
        token="token_456",
        path_in_repo="custom/sub-path.jsonl",
        private=True,
        commit_message="Upload",
    )

    dataset_instance.push_to_hub.assert_called_once_with(
        repo_id="tester/dataset",
        token="token_456",
        private=True,
        commit_message="Upload",
        split="custom_sub_path",
    )

