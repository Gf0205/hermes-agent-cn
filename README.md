# Hermes Agent CN

Hermes Agent CN is a production-grade, Hermes-inspired AI agent with a layered architecture for planning, execution, memory, recall, and closed-loop verification.

## Key features
- Strategic + tactical planning (structured sub-goals)
- Robust execution loop (checkpoint / resume, parallel tools with fallback)
- Context compression (ContextCompressor v2 with anti-thrashing and cooldown)
- Procedural skills (auto-distillation + governance with rollback)
- Cross-session recall (hybrid lexical/semantic/recency + explainable match reasons)
- Offline evaluation + regression gates (Phase 6-9 verify scripts)
- Online observability (recall health + JSONL logging)

## Requirements
- Python 3.10+

## Setup
1. Copy environment template:
   - `cp .env.example .env` (edit values)
2. Install dependencies:
   - `pip install -e .`
   - (optional) `pip install -e .[dev]`

## Run
- Interactive (with TUI):
  - `python src/main.py --tui`
- Non-interactive:
  - `python src/main.py --goal "your task here" --context ""`

## Verify (regression)
Run the full Phase 9 recall health gate (recommended):
- `python verify/step34_phase9.py`

## Notes (Windows / Chroma)
- If Chroma initialization is blocked by Windows policy, the project degrades to an in-memory semantic store so the main agent can still run.

## License
MIT

