# Python environment and dependencies

- Use uv for dependency management and package installation.
- Use the project virtual environment at `.venv`; run Python commands through
  `uv run` so they use this environment.
- Add required packages with `uv add <package>` and development dependencies with
  `uv add --dev <package>`. Keep `pyproject.toml` and `uv.lock` in sync.
- Restore the environment with `uv sync --locked`.
- The uv project currently covers the causal benchmark. The original foundation
  model pipeline has a separate dependency specification in `environment.yml`;
  migrate its dependencies deliberately when that pipeline is needed.

# Downstream research scripts

- Before implementing new downstream scripts, inspect the relevant SHRUG-FM
  code/configuration for reusable components and check scientific/data assumptions.
- Reuse working components where appropriate, and document dataset, preprocessing,
  label-support, architecture, and evaluation differences from SHRUG-FM.
- Preserve original split/event identifiers and keep invalid pixels out of both
  prediction and reference fractions on explicitly matched support.
