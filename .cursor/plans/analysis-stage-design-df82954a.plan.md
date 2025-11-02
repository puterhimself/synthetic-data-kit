<!-- df82954a-6b39-4ff9-8bc4-94d7d0da1cb4 80c45041-5eb5-406a-8849-ec1cf2ddc4af -->
# Analysis Stage for syntheticdatakit

## 1) High-level overview

- The analysis stage runs automatically before existing flows (`ingest`, `create`, `curate`) whenever the user provides an input path (file or directory) via CLI or server, unless explicitly skipped.
- The stage can also be invoked directly as `sdkit analyze <path>`; it emits a JSON report (to stdout or `--output` path) summarizing insights for all discovered files and persists a cache file (default `.sdkit/analysis/report.json`).
- It extracts reusable text insights: format detection, language, summaries, keywords/tags, categorization, size/token counts, and quality/PII heuristics. Results are structured via shared dataclasses and serializable to JSON.
- Configuration is shared between CLI/server through typed config objects loaded from existing YAML (`synthetic_data_kit/config.yaml`, `configs/config.yaml`); CLI flags override config; server payload overrides both.

### Proposed API surface (consistent with current patterns)

- New dataclasses in `synthetic_data_kit/core/analysis_types.py`:
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class AnalysisOptions:
    use_llm: bool = False
    max_chars: int = 100_000
    keyword_top_k: int = 15
    categorize: bool = True
    detect_language: bool = True
    detect_pii: bool = False
    cache_enabled: bool = True
    output_path: Optional[Path] = None
    persist_cache: bool = True

@dataclass
class FileAnalysis:
    path: Path
    format: str
    language: Optional[str]
    char_count: int
    token_count: int
    summary: Optional[str]
    keywords: List[str]
    tags: List[str]
    category: Optional[str]
    warnings: List[str]
    extraction_time_ms: float
    analysis_time_ms: float

@dataclass
class AnalysisReport:
    root: Path
    files: List[FileAnalysis]
    aggregate: Dict[str, object]  # e.g., top keywords/tags, categories histogram
    stats: Dict[str, object]      # e.g., time_ms, num_files, fail_count, cache_hits
```

- Config types in `synthetic_data_kit/utils/config_types.py` to ensure typed access:
```python
@dataclass
class AnalysisConfig:
    enabled: bool = True
    use_llm: bool = False
    max_chars: int = 100_000
    keyword_top_k: int = 15
    categorize: bool = True
    detect_language: bool = True
    detect_pii: bool = False
    cache: bool = True
    default_output: str = ".sdkit/analysis/report.json"
```

- `AnalysisConfig` integrates into `synthetic_data_kit/utils/config.py` loader and is exposed via `Config` root dataclass.
- Orchestrator in `synthetic_data_kit/core/analysis.py`:
```python
def analyze_path(input_path: Path, options: AnalysisOptions, context: Optional[Context] = None) -> AnalysisReport: ...
```

- CLI command/flags in `synthetic_data_kit/cli.py`: `sdkit analyze <path>`, `--output`, `--analysis-only`, `--skip-analysis`, `--no-llm`, `--max-files`, `--max-chars`, `--no-cache`.
- Server: `POST /analysis` (start), `GET /analysis?path=...` (read/cached), `GET /analysis/{job_id}` (progress), plus template for report preview.

## 2) Phased task breakdown (ordered)

### Phase A — Types, config, and plumbing (P0)

1. Define analysis dataclasses and serialization helpers

   - Intent: Establish shared contracts serializable to JSON for CLI/server.
   - Affected: `synthetic_data_kit/core/analysis_types.py`, `synthetic_data_kit/core/__init__.py` (exports), `synthetic_data_kit/utils/json.py` (if helper needed).
   - Complexity: S, Priority: P0.

2. Add typed config objects and defaults

   - Intent: Extend `Config` structure with `AnalysisConfig`, load from `synthetic_data_kit/config.yaml` and `configs/config.yaml`, support env overrides.
   - Affected: `synthetic_data_kit/utils/config_types.py` (new), `synthetic_data_kit/utils/config.py`, YAML files mentioned.
   - Complexity: M, Priority: P0.

3. Config serialization to CLI options

   - Intent: Map config defaults to `AnalysisOptions`, allowing CLI flags to override.
   - Affected: `synthetic_data_kit/cli.py`, `synthetic_data_kit/core/context.py` (if storing config).
   - Complexity: S, Priority: P0.

### Phase B — Core local analysis capabilities (P0)

4. Reuse/extend format detection and text extraction

   - Intent: Uniformly detect file type and extract text via parsers; handle unsupported files with warnings.
   - Affected: `synthetic_data_kit/parsers/*.py`, `synthetic_data_kit/utils/directory_processor.py`, `synthetic_data_kit/utils/text.py`.
   - Complexity: M, Priority: P0.

5. Local summary heuristic

   - Intent: Provide fast baseline summaries (lead sentences, frequency scoring) respecting `max_chars`.
   - Affected: `synthetic_data_kit/utils/text_analysis.py` (new), `synthetic_data_kit/utils/text.py` (helpers).
   - Complexity: M, Priority: P0.

6. Keyword/tag extraction

   - Intent: Extract top n-grams with stopword filtering; map to tags.
   - Affected: `synthetic_data_kit/utils/text_analysis.py`.
   - Complexity: M, Priority: P0.

7. Categorization via rules

   - Intent: Map files to categories using format + keyword heuristics; configurable taxonomy.
   - Affected: `synthetic_data_kit/utils/text_analysis.py`, potential taxonomy in YAML (`configs/analysis_taxonomy.yaml` optional).
   - Complexity: S, Priority: P0.

8. Language detection and PII heuristics

   - Intent: Provide language guesses and regex-based PII warnings.
   - Affected: `synthetic_data_kit/utils/text_analysis.py`.
   - Complexity: S, Priority: P0.

### Phase C — Orchestration, parallelism, aggregation, and persistence (P0)

9. Implement `analyze_path`

   - Intent: Enumerate files (respecting `max_files`), run per-file analysis with thread pool, collect timing metadata.
   - Affected: `synthetic_data_kit/core/analysis.py`, `synthetic_data_kit/utils/directory_processor.py` (filters), `synthetic_data_kit/core/context.py`.
   - Complexity: M, Priority: P0.

10. Aggregate insights and metrics

   - Intent: Build report-wide stats (top keywords/tags, categories counts, warnings list).
   - Affected: `synthetic_data_kit/core/analysis.py`.
   - Complexity: S, Priority: P0.

11. JSON serialization & default output

   - Intent: Serialize `AnalysisReport` to JSON and write to default output path or CLI `--output`; ensure deterministic structure.
   - Affected: `synthetic_data_kit/core/analysis.py`, `synthetic_data_kit/utils/json.py` (helper), `synthetic_data_kit/utils/filesystem.py` if exists.
   - Complexity: S, Priority: P0.

12. Optional caching (content fingerprint)

   - Intent: Skip unchanged files using mtime/size hash; store per-file JSON under `.sdkit/analysis/cache/` when enabled via config.
   - Affected: `synthetic_data_kit/core/analysis.py`, new `synthetic_data_kit/utils/cache.py`.
   - Complexity: M, Priority: P1.

### Phase D — CLI integration (P0)

13. Add explicit `sdkit analyze` command

   - Intent: Provide command with `--output`, `--format json|yaml`, `--no-cache`, `--config CONFIG_PATH` support.
   - Affected: `synthetic_data_kit/cli.py`, possibly `synthetic_data_kit/utils/io.py` for output writers.
   - Complexity: M, Priority: P0.

14. Pre-execution hook for existing flows

   - Intent: Invoke analysis stage before `create/curate/ingest` when config `analysis.enabled` is true; allow `--skip-analysis` or config opt-out.
   - Affected: `synthetic_data_kit/cli.py`, `synthetic_data_kit/core/create.py`, `curate.py`, `ingest.py` (accept report in context).
   - Complexity: S, Priority: P0.

### Phase E — Optional LLM enhancement (P1)

15. Integrate `models/llm_client.py`

   - Intent: Respect `options.use_llm`; generate enhanced summaries/tags when enabled; log cost/latency.
   - Affected: `synthetic_data_kit/core/analysis.py`, `synthetic_data_kit/utils/llm_processing.py`, `synthetic_data_kit/models/llm_client.py`.
   - Complexity: M, Priority: P1.

### Phase F — Server integration (P1)

16. REST + background job for analysis

   - Intent: Add endpoints to trigger analysis, poll status, download report file; reuse orchestrator.
   - Affected: `synthetic_data_kit/server/app.py`, `synthetic_data_kit/server/templates/analysis.html` (new).
   - Complexity: M, Priority: P1.

17. Server config overrides

   - Intent: Allow request payload to override defaults (`use_llm`, `max_chars`), fallback to `AnalysisConfig`.
   - Affected: `synthetic_data_kit/server/app.py`.
   - Complexity: S, Priority: P1.

### Phase G — Tests and docs (P0/P1)

18. Unit tests for heuristics and serialization

   - Intent: Cover `text_analysis` functions, config loading, JSON output schema.
   - Affected: `tests/unit/test_text_analysis.py` (new), extend `tests/unit/test_config.py` if present or create.
   - Complexity: M, Priority: P0.

19. Functional tests for CLI

   - Intent: Verify `sdkit analyze` command, output file creation, config overrides, skip/only behavior.
   - Affected: `tests/functional/test_analysis_cli.py` (new), extend `tests/functional/test_cli.py`.
   - Complexity: M, Priority: P0.

20. Integration tests with flows

   - Intent: Confirm analysis runs once before flows, respects config toggles, ensures downstream unaffected.
   - Affected: `tests/integration/test_analysis_flow.py` (new).
   - Complexity: M, Priority: P1.

21. Server functional tests

   - Intent: Validate API endpoints, background job, report download.
   - Affected: `tests/functional/test_analysis_server.py` (new).
   - Complexity: M, Priority: P1.

22. Documentation updates

   - Intent: Document command usage, config options, sample JSON output, server endpoints.
   - Affected: `README.md`, `DOCS.md`, `use-cases/getting-started/README.md`, new `docs/analysis.md` optional.
   - Complexity: S, Priority: P0.

## 3) Task details summary

- For each task above: intent, affected modules, complexity, priority already specified for quick reference.

## 4) Risks, open questions, dependencies

- LLM latency/cost: keep disabled by default; config flag to opt-in per environment.
- Large directories: enforce `max_files`, streaming progress logs, background job on server.
- Parser gaps: ensure every supported file type implements `extract_text`; add graceful fallback with warnings.
- Config coherence: Need to confirm existing `Config` dataclass shape to integrate `AnalysisConfig`; may require migration for users.
- Cache invalidation: choose fingerprint approach (mtime+size vs hash); consider user-provided thresholds in config.
- Output size: For very large analyses, ensure JSON file manageable (maybe chunk or compress?).
- Thread safety: Guarantee parsers and LLM client safe for concurrent use; limit thread pool via config.
- Taxonomy governance: Keep small initial taxonomy; document extension mechanism in `config.yaml`.

Dependencies to validate

- Confirm `synthetic_data_kit/utils/config.py` uses pydantic/dataclass or custom loader; adjust to support nested typed configs.
- Verify standard library or existing dependencies include language detection; otherwise add lightweight lib (e.g., `langdetect`) with optional extra (document requirement).

## 5) Testing strategy

- Unit: text normalization, keyword extraction, categorization, language/PII heuristics, config load/dump, JSON serialization round-trip.
- Functional CLI: run `sdkit analyze tests/data --output tmp.json`, ensure file exists, schema valid, flags override config, skip/only toggles operate.
- Integration: full `create` flow with analysis on/off; ensure context receives report and results persisted.
- Server: call `/analysis` endpoints, confirm background job status, verify downloaded JSON matches schema.
- Performance: evaluate directories with 100–500 files; check execution time, memory, caching effectiveness.

## 6) Metrics and logging

- Per-file metrics: `extraction_time_ms`, `analysis_time_ms`, `char_count`, `token_count`, `truncated`, `llm_used`, `warnings` (structured).
- Aggregate metrics: `total_time_ms`, `num_files_analyzed`, `num_skipped`, `num_errors`, `cache_hits`, `categories_histogram`, `top_keywords`.
- Logging: INFO for lifecycle (start, progress, finish with totals), DEBUG for per-file details when enabled, WARN for parser failures/unsupported formats; ensure logs include path/correlation id.
- Persisted report metadata: embed generation timestamp, config hash, CLI command/version in JSON header for traceability.

## Notes on reusability and consistency

- `analyze_path()` is the single entry point used by both CLI command and server background task.
- `AnalysisOptions` built from `AnalysisConfig` + overrides ensures consistent behavior across surfaces.
- JSON schema exported via `AnalysisReport` dataclass to keep CLI/server outputs interchangeable; consider version field for future evolution.

### To-dos

- [ ] Add analysis types and config defaults
- [ ] Wire parsers to consistent extract_text and normalize text
- [ ] Implement local summary, keyword, category, language, PII heuristics
- [ ] Create analyze_path orchestrator with parallelism and aggregation
- [ ] Add sdkit analyze command and flags
- [ ] Invoke analysis before create/curate/ingest with skip/only flags
- [ ] Add optional LLM-enhanced summary/tags/categorization
- [ ] Implement cache keys and persist analysis.json
- [ ] Add FastAPI endpoints and background job for analysis
- [ ] Add analysis preview template and link from files view
- [ ] Add unit and functional tests for analysis and CLI
- [ ] Add integration tests with existing flows
- [ ] Update README/DOCS/use-cases with analysis examples