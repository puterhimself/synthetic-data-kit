# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Analysis orchestrator for file and directory analysis
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import yaml

from synthetic_data_kit.types.analysis_types import AnalysisOptions, FileAnalysis, AnalysisReport
from synthetic_data_kit.core.ingest import determine_parser
from synthetic_data_kit.utils.text_analysis import (
    extract_text_from_parser_result,
    normalize_text,
    estimate_token_count,
    generate_summary,
    extract_keywords,
    extract_tags,
    categorize_file,
    detect_language,
    detect_pii,
    analyze_text_quality,
    compute_readability_score,
    detect_tone,
    detect_register,
)
from synthetic_data_kit.utils.coverage_labeler import label_coverage
from synthetic_data_kit.utils.dedup import compute_minhash
from synthetic_data_kit.utils.lance_utils import create_lance_dataset
from synthetic_data_kit.utils.directory_processor import get_supported_files, INGEST_EXTENSIONS

logger = logging.getLogger(__name__)


def analyze_single_file(
    file_path: Path, options: AnalysisOptions, config: Optional[Dict[str, Any]] = None
) -> FileAnalysis:
    """Analyze a single file"""
    extraction_start = time.time()
    warnings = []

    try:
        # Determine parser
        parser = determine_parser(str(file_path), config or {}, multimodal=False)

        # Extract text
        parser_result = parser.parse(str(file_path))
        text = extract_text_from_parser_result(parser_result)

        extraction_time = (time.time() - extraction_start) * 1000  # Convert to ms

        # Truncate if needed
        truncated = False
        if len(text) > options.max_chars:
            text = text[: options.max_chars]
            truncated = True
            warnings.append("truncated")

        # Analyze text
        analysis_start = time.time()

        # Basic metrics
        char_count = len(text)
        token_count = estimate_token_count(text)

        # Language detection
        language = None
        if options.detect_language:
            language = detect_language(text)

        # Summary
        summary = None
        if options.use_llm:
            # LLM-enhanced summary (to be implemented in Phase E)
            summary = generate_summary(text)  # Fallback for now
        else:
            summary = generate_summary(text)

        # Keywords and tags
        keywords = extract_keywords(text, top_k=options.keyword_top_k)
        tags = extract_tags(text, keywords)

        # Categorization
        category = None
        if options.categorize:
            category = categorize_file(file_path, text, keywords)

        # PII detection
        safety_warnings = []
        if options.detect_pii:
            pii_warnings = detect_pii(text)
            warnings.extend(pii_warnings)
            safety_warnings.extend(pii_warnings)

        # Quality checks
        quality_warnings = analyze_text_quality(text)
        warnings.extend(quality_warnings)

        # Extended analysis features
        analysis_config = config.get("analysis", {}) if config else {}

        # Coverage labeling
        coverage = None
        if analysis_config.get("coverage", {}).get("enabled", True):
            keyword_rules = analysis_config.get("coverage", {}).get("keyword_rules", {})
            llm_fallback = analysis_config.get("coverage", {}).get("llm_fallback", True)
            # Only use LLM if enabled globally
            llm_client = None
            if llm_fallback and options.use_llm:
                tmp_config_path: Optional[Path] = None
                try:
                    from synthetic_data_kit.models.llm_client import LLMClient

                    if config:
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=".yaml", delete=False
                        ) as tmp_file:
                            yaml.safe_dump(config, tmp_file)
                            tmp_config_path = Path(tmp_file.name)
                        llm_client = LLMClient(config_path=tmp_config_path)
                    else:
                        llm_client = LLMClient(config_path=None)
                except Exception as exc:
                    logger.debug(
                        "Could not initialize LLM client for coverage labeling: %s",
                        exc,
                    )
                    llm_client = None
                finally:
                    if tmp_config_path and tmp_config_path.exists():
                        try:
                            os.unlink(tmp_config_path)
                        except OSError:
                            logger.debug(
                                "Failed to remove temporary config file: %s",
                                tmp_config_path,
                            )

            coverage = label_coverage(
                text, keywords, keyword_rules, llm_fallback, llm_client, config
            )

        # Difficulty analysis
        difficulty = None
        if analysis_config.get("difficulty", {}).get("enabled", True):
            if analysis_config.get("difficulty", {}).get("readability_bands", True):
                difficulty = compute_readability_score(text)

        # Style analysis
        tone = None
        register = None
        if analysis_config.get("style", {}).get("enabled", True):
            if analysis_config.get("style", {}).get("tone_detection", True):
                tone = detect_tone(text)
            if analysis_config.get("style", {}).get("register_detection", True):
                register = detect_register(text)

        # Dedup signature
        duplicate_signature = None
        if analysis_config.get("dedup", {}).get("enabled", True):
            if analysis_config.get("dedup", {}).get("minhash_lsh", True):
                minhash_sig = compute_minhash(text)
                # Store signature as string representation
                duplicate_signature = ",".join(
                    map(str, minhash_sig.hash_values[:16])
                )  # Store first 16 for compactness

        analysis_time = (time.time() - analysis_start) * 1000  # Convert to ms

        # Determine file format
        file_format = file_path.suffix.lower() or "unknown"

        return FileAnalysis(
            path=file_path,
            format=file_format,
            language=language,
            char_count=char_count,
            token_count=token_count,
            summary=summary,
            keywords=keywords,
            tags=tags,
            category=category,
            warnings=warnings,
            extraction_time_ms=extraction_time,
            analysis_time_ms=analysis_time,
            truncated=truncated,
            llm_used=options.use_llm,
            coverage=coverage,
            difficulty=difficulty,
            safety_warnings=safety_warnings,
            tone=tone,
            register=register,
            duplicate_signature=duplicate_signature,
        )

    except Exception as e:
        logger.warning(f"Error analyzing {file_path}: {e}")
        extraction_time = (time.time() - extraction_start) * 1000
        return FileAnalysis(
            path=file_path,
            format=file_path.suffix.lower() or "unknown",
            warnings=[f"error: {str(e)}"],
            extraction_time_ms=extraction_time,
            analysis_time_ms=0.0,
        )


def aggregate_insights(file_analyses: List[FileAnalysis]) -> Dict[str, Any]:
    """Aggregate insights from multiple file analyses"""
    aggregate = {
        "top_keywords": [],
        "top_tags": [],
        "categories": {},
        "languages": {},
        "warnings": [],
    }

    # Collect all keywords and tags
    all_keywords = []
    all_tags = []
    category_counts = {}
    language_counts = {}
    all_warnings = []

    for analysis in file_analyses:
        all_keywords.extend(analysis.keywords)
        all_tags.extend(analysis.tags)

        if analysis.category:
            category_counts[analysis.category] = category_counts.get(analysis.category, 0) + 1

        if analysis.language:
            language_counts[analysis.language] = language_counts.get(analysis.language, 0) + 1

        all_warnings.extend(analysis.warnings)

    # Get top keywords
    from collections import Counter

    keyword_counter = Counter(all_keywords)
    aggregate["top_keywords"] = [word for word, _ in keyword_counter.most_common(20)]

    # Get top tags
    tag_counter = Counter(all_tags)
    aggregate["top_tags"] = [tag for tag, _ in tag_counter.most_common(15)]

    # Categories histogram
    aggregate["categories"] = dict(
        sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Languages histogram
    aggregate["languages"] = dict(sorted(language_counts.items(), key=lambda x: x[1], reverse=True))

    # Unique warnings
    aggregate["warnings"] = list(set(all_warnings))

    return aggregate


def analyze_path(
    input_path: Path, options: AnalysisOptions, config: Optional[Dict[str, Any]] = None
) -> AnalysisReport:
    """Analyze a file or directory path"""
    start_time = time.time()

    # Resolve path
    input_path = Path(input_path).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Path does not exist: {input_path}")

    file_analyses = []

    if input_path.is_file():
        # Single file analysis
        logger.info(f"Analyzing file: {input_path}")
        analysis = analyze_single_file(input_path, options, config)
        file_analyses = [analysis]
        root_path = input_path.parent
    else:
        # Directory analysis
        logger.info(f"Analyzing directory: {input_path}")

        # Get supported files
        supported_files = get_supported_files(str(input_path), INGEST_EXTENSIONS)

        # Apply max_files limit if specified
        if options.max_files and len(supported_files) > options.max_files:
            supported_files = supported_files[: options.max_files]
            logger.info(f"Limited to {options.max_files} files")

        logger.info(f"Found {len(supported_files)} files to analyze")

        # Process files in parallel
        max_workers = min(8, len(supported_files))  # Limit parallelism
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(analyze_single_file, Path(f), options, config): f
                for f in supported_files
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Processed {completed}/{len(supported_files)} files")

                try:
                    analysis = future.result()
                    file_analyses.append(analysis)
                except Exception as e:
                    file_path = futures[future]
                    logger.error(f"Error processing {file_path}: {e}")
                    file_analyses.append(
                        FileAnalysis(
                            path=Path(file_path),
                            format=Path(file_path).suffix.lower() or "unknown",
                            warnings=[f"processing_error: {str(e)}"],
                            extraction_time_ms=0.0,
                            analysis_time_ms=0.0,
                        )
                    )

        root_path = input_path

    # Aggregate insights
    aggregate = aggregate_insights(file_analyses)

    # Calculate statistics
    total_time = (time.time() - start_time) * 1000  # Convert to ms
    stats = {
        "total_time_ms": total_time,
        "num_files_analyzed": len(file_analyses),
        "num_errors": sum(1 for f in file_analyses if any("error" in w for w in f.warnings)),
        "cache_hits": 0,  # Will be implemented in Phase E
        "total_chars": sum(f.char_count for f in file_analyses),
        "total_tokens": sum(f.token_count for f in file_analyses),
        "avg_extraction_time_ms": sum(f.extraction_time_ms for f in file_analyses)
        / len(file_analyses)
        if file_analyses
        else 0,
        "avg_analysis_time_ms": sum(f.analysis_time_ms for f in file_analyses) / len(file_analyses)
        if file_analyses
        else 0,
    }

    # Create report
    report = AnalysisReport(root=root_path, files=file_analyses, aggregate=aggregate, stats=stats)

    # Persist if requested
    if options.persist_cache and options.output_path:
        report.save(options.output_path)
        logger.info(f"Saved analysis report to {options.output_path}")

        # Write Lance dataset if enabled
        analysis_config = config.get("analysis", {}) if config else {}
        if analysis_config.get("dedup", {}).get("enabled", True):
            lance_path = str(Path(options.output_path).with_suffix(".lance"))
            rows = [fa.to_dict() for fa in file_analyses]
            try:
                create_lance_dataset(rows, lance_path)
                logger.info(f"Saved Lance dataset to {lance_path}")
            except Exception as e:
                logger.warning(f"Failed to write Lance dataset: {e}")

    return report
