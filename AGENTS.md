# AGENTS.md

Project-level instructions for Codex and other coding agents working in this repository.

## Python Environment

Use uv and the project `.venv` for Python commands:

```bash
uv run --offline --no-sync python ...
```

Download missing packages only on a newly created Inspire CPU Notebook. Run
`uv sync` there; GPU workers use `uv sync --offline --locked` or `uv run --no-sync`.

## Persistent Session Management

For long-running training jobs and interactive debugging, use `zmx` to manage
persistent sessions — **do not use `tmux`**. `zmx` is the project's standard
session manager for training and debug persistence.

