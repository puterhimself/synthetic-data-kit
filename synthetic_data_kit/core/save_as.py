# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Logic for saving file format

import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from synthetic_data_kit.utils.format_converter import (
    to_jsonl,
    to_alpaca,
    to_fine_tuning,
    to_chatml,
    to_hf_dataset,
    to_conversations_jsonl,
)
from synthetic_data_kit.utils.llm_processing import convert_to_conversation_format


def convert_format(
    input_path: str,
    output_path: str,
    format_type: str,
    config: Optional[Dict[str, Any]] = None,
    storage_format: str = "json",
) -> str:
    """Convert data to different formats

    Args:
        input_path: Path to the input file
        output_path: Path to save the output
        format_type: Output format (jsonl, alpaca, ft, chatml)
        config: Configuration dictionary
        storage_format: Storage format, either "json" or "hf" (Hugging Face dataset)

    Returns:
        Path to the output file or directory
    """
    # Load input file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract data based on known structures
    # Try to handle the case where we have QA pairs or conversations
    if "qa_pairs" in data:
        qa_pairs = data.get("qa_pairs", [])
    elif "filtered_pairs" in data:
        qa_pairs = data.get("filtered_pairs", [])
    elif "conversations" in data:
        conversations = data.get("conversations", [])
        qa_pairs = []
        for conv in conversations:
            if len(conv) >= 3 and conv[1]["role"] == "user" and conv[2]["role"] == "assistant":
                qa_pairs.append({"question": conv[1]["content"], "answer": conv[2]["content"]})
    else:
        # If the file is just an array of objects, check if they look like QA pairs
        if isinstance(data, list):
            qa_pairs = []
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    qa_pairs.append(item)
        else:
            raise ValueError("Unrecognized data format - expected QA pairs or conversations")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # When using HF dataset storage format
    if storage_format == "hf":
        # For HF datasets, we need to prepare the data in the right structure
        if format_type == "jsonl":
            # For JSONL, just use the QA pairs directly
            formatted_pairs = qa_pairs
        elif format_type == "alpaca":
            # Format as Alpaca structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append(
                    {"instruction": pair["question"], "input": "", "output": pair["answer"]}
                )
        elif format_type == "ft":
            # Format as OpenAI fine-tuning structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append(
                    {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]},
                        ]
                    }
                )
        elif format_type == "chatml":
            # Format as ChatML structure
            formatted_pairs = []
            for pair in qa_pairs:
                formatted_pairs.append(
                    {
                        "messages": [
                            {"role": "system", "content": "You are a helpful AI assistant."},
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]},
                        ]
                    }
                )
        elif format_type == "conversation":
            conversations = convert_to_conversation_format(qa_pairs)
            formatted_pairs = [
                {"conversations": conversation} for conversation in conversations
            ]
        else:
            raise ValueError(f"Unknown format type: {format_type}")

        # Save as HF dataset (Arrow format)
        return to_hf_dataset(formatted_pairs, output_path)

    # Standard JSON file storage format
    else:
        # Convert to the requested format using existing functions
        if format_type == "jsonl":
            return to_jsonl(qa_pairs, output_path)
        elif format_type == "alpaca":
            return to_alpaca(qa_pairs, output_path)
        elif format_type == "ft":
            return to_fine_tuning(qa_pairs, output_path)
        elif format_type == "chatml":
            return to_chatml(qa_pairs, output_path)
        elif format_type == "conversation":
            conversations = convert_to_conversation_format(qa_pairs)
            return to_conversations_jsonl(conversations, output_path)
        else:
            raise ValueError(f"Unknown format type: {format_type}")


def push_to_huggingface_hub(
    output_path: str,
    repo_id: str,
    token: str,
    *,
    private: bool = False,
    path_in_repo: Optional[str] = None,
    commit_message: Optional[str] = None,
) -> str:
    """Upload a converted artifact to the Hugging Face Hub.

    Args:
        output_path: File or directory generated by `convert_format`.
        repo_id: Target dataset repository identifier (e.g., "user/dataset").
        token: Authentication token for Hugging Face Hub.
        private: Whether to keep the repository private.
        path_in_repo: Optional path within the repository to place the artifact.
        commit_message: Optional custom commit message.

    Returns:
        URL of the updated Hugging Face Hub repository or artifact.
    """
    if not repo_id:
        raise ValueError("Hugging Face repo_id is required to push artifacts")
    if not token:
        raise ValueError("Hugging Face token is required to push artifacts")

    try:
        from huggingface_hub import HfApi  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'huggingface_hub' package is required to push artifacts. Install it with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    commit_kwargs: Dict[str, Any] = {}
    if commit_message:
        commit_kwargs["commit_message"] = commit_message

    abs_output = os.path.abspath(output_path)

    if os.path.isdir(abs_output):
        dataset_indicator_files = [
            os.path.join(abs_output, "dataset_info.json"),
            os.path.join(abs_output, "dataset_dict.json"),
            os.path.join(abs_output, "state.json"),
        ]

        if any(os.path.exists(path) for path in dataset_indicator_files):
            try:
                from datasets import load_from_disk  # type: ignore[import]
            except ImportError as exc:
                raise ImportError(
                    "The 'datasets' package is required to push saved datasets. Install it with: pip install datasets"
                ) from exc

            dataset = load_from_disk(abs_output)
            push_kwargs: Dict[str, Any] = {"repo_id": repo_id, "token": token, "private": private}
            if commit_message:
                push_kwargs["commit_message"] = commit_message

            dataset.push_to_hub(**push_kwargs)
            return f"https://huggingface.co/datasets/{repo_id}"

        target_path = path_in_repo or os.path.basename(abs_output)
        if not target_path or target_path == ".":
            target_path = "."

        api.upload_folder(
            folder_path=abs_output,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=target_path,
            **commit_kwargs,
        )
        return f"https://huggingface.co/datasets/{repo_id}"


    from datasets import Dataset

    def _sanitize_split(name: str) -> str:
        base = os.path.splitext(name)[0]
        sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
        return sanitized if sanitized else "train"

    dataset = Dataset.from_json(abs_output)
    split_name = _sanitize_split(path_in_repo or os.path.basename(abs_output))
    dataset.push_to_hub(
        repo_id=repo_id,
        token=token,
        private=private,
        commit_message=commit_message,
        split=split_name,
    )
    return f"https://huggingface.co/datasets/{repo_id}"
