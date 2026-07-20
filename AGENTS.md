# AGENTS.md

Project-level instructions for Codex and other coding agents working in this repository.

## Python Environment

Use the project Conda environment for all Python commands:

```bash
conda activate qwen-inpainting-pre-insertion
```

## Persistent Session Management

For long-running training jobs and interactive debugging, use `zmx` to manage
persistent sessions — **do not use `tmux`**. `zmx` is the project's standard
session manager for training and debug persistence.

## Web Search

Prefer **Exa MCP** for web searches instead of the built-in `WebSearch` / `WebFetch` tools.

When calling `web_fetch_exa`, use a large `maxCharacters` (e.g. 50000) by default to avoid truncated results.

## 