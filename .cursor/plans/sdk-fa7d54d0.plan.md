<!-- fa7d54d0-65fc-452b-bd72-b34660fecb70 51e2bdf0-e160-4274-8311-966f63539d0b -->
# Analyze + Plan + Plan‑driven Generate for Synthetic Data Kit

## Scope

Insert Analyze and Plan between ingest and create, then make create honor plan.json for deterministic quotas, prompts, and gates. Keep CLI/YAML compatibility and Lance outputs.

## Changes

- CLI
  - Add `plan` command: `synthetic-data-kit plan --analysis analysis.json -c planning.yaml -o plan.json`.
  - Extend `create` to accept `--plan plan.json` and `--dry-run`; add `generate` alias.
- Analyze (hybrid labeler per choice 1a+1c)
  - Coverage: keyword rules first; LLM fallback for low-confidence/unlabeled.
  - Difficulty: readability bands + optional judge score.
  - Safety: regex PII, policy/blocklist matching; counters.
  - Dupes (choice 2c): MinHash LSH first, embedding tie-break at cosine≥0.92.
  - Style: tone/register heuristics.
  - Outputs: report.json + `.lance` rows (per-file metrics, labels, sketches).
- Plan
  - Build plan.json from analysis + planning.yaml: dataset_goal, mix quotas, coverage/difficulty/style targets, sampler, prompt families/rotation, safety/dedup rules, gates, budget.
  - Deterministic prompt rotation and per-task sampler overrides.
- Generation
  - When `--plan` present: compute per-task quotas; schedule per-file batches; apply sampler/prompt families; enforce dedup pre-check; support `--dry-run` counts; emit progress and residual quotas.
  - Pass plan min_quality to curate by default (or print run-next command with threshold).
- PromptComponents
  - Add structured binding into prompt templates: inject `{taskContext}`, `{toneContext}`, `{backgroundData}`, `{examples}`, `{finalRequest}`, `{outputFormatting}`, etc.

## File-level work

- Add: `synthetic_data_kit/core/plan.py` (build_plan), `synthetic_data_kit/core/planning_types.py` (PlanSpec), `synthetic_data_kit/utils/dedup.py` (minhash + embed tie-break), `synthetic_data_kit/utils/coverage_labeler.py` (keyword+LLM hybrid).
- Edit: `synthetic_data_kit/cli.py` (new plan cmd; create --plan/--dry-run; generate alias), `synthetic_data_kit/core/analysis.py` (detectors; Lance write), `synthetic_data_kit/core/analysis_types.py` (fields), `synthetic_data_kit/core/create.py` (consume plan; quotas; prompts), `synthetic_data_kit/utils/config.py` (get_plan_config), `synthetic_data_kit/generators/qa_generator.py` (PromptComponents binding; prompt families).
- Config: extend `synthetic_data_kit/config.yaml` with analyze/plan sections.

## Tiny code shape (reference)

```12:24:synthetic_data_kit/cli.py
@app.command("plan")
def plan(analysis: Path = typer.Option(..., "--analysis", "-a"),
        output: Path = typer.Option("data/analysis/plan.json", "--output", "-o"),
        config: Optional[Path] = typer.Option(None, "--config", "-c")):
    cfg = load_config(config or ctx.config_path)
    from synthetic_data_kit.core.plan import build_plan
    spec = build_plan(analysis, cfg)
    spec.save(output)
    console.print(f"✅ Plan saved to {output}")
```
```181:207:synthetic_data_kit/core/analysis.py
# after aggregate
# write Lance rows
from synthetic_data_kit.utils.lance_utils import create_lance_dataset
rows = [fa.to_dict() for fa in file_analyses]
create_lance_dataset(rows, str(Path(options.output_path).with_suffix('.lance')))
```

### To-dos

- [ ] Add analyze/plan sections to config.yaml
- [ ] Extend analysis with coverage/difficulty/safety/style and Lance write
- [ ] Implement MinHash LSH and optional embedding tie-break
- [ ] Add keyword rules + LLM fallback labeler
- [ ] Create PlanSpec dataclasses (planning_types.py)
- [ ] Build plan.json from analysis + planning.yaml
- [ ] Add plan command to cli.py
- [ ] Make create accept --plan/--dry-run and enforce quotas/prompts
- [ ] Add generate alias that calls create with --plan
- [ ] Bind PromptComponents into prompt templates in qa_generator
- [ ] Add unit tests for labeler, dedup, plan builder
- [ ] Integration test: ingest→analyze→plan→create --plan→curate