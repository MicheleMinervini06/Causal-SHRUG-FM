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

# Code simplicity

- Prefer the smallest clear implementation that satisfies the current task.
  Every function and line should serve a concrete purpose; remove unused code,
  redundant logic, and speculative abstractions instead of growing the codebase.
- Reuse existing helpers where they fit. Add new abstractions or dependencies
  only when they materially simplify the implementation or are necessary for
  correctness. Keep scientific assumptions and meaningful checks explicit.

# Project research note

- Frame open questions around what we want to learn. A question should still
  make sense if the dataset, method, or budget changes.
- When we choose a question, describe the specific comparison, budget, and
  success measures in the experiment. Put agreed actions in `Next steps`.
- Keep `C:\Users\miner003\OneDrive - Wageningen University & Research\Documenten\Obsidian Vault\6 - Main Note\Projects\Causal-UQ.md`
  updated as substantive project results, decisions, limitations and open paths change.
- Update the relevant experiment and next steps concisely. Preserve unrelated
  sections; refer to detailed repository reports instead of duplicating tables
  or routine logs in the note.

# Local development and Anunna execution

- Develop and review code locally in this workspace. Run small implementation
  checks locally; use Anunna for GPU checks and substantial training/evaluation.
- Synchronize source code, Slurm scripts, `pyproject.toml`, and `uv.lock` through
  the user's Git repository (`origin`); keep the original SHRUG-FM repository as
  `upstream`. Commit/push only when authorized; do not assume local edits have
  already reached GitHub or Anunna.
- On Anunna, use the project on assigned Lustre storage. Its current location is
  `/lustre/nobackup/WUR/AIN/miner003/SHRUG-FM`. Preserve existing data, cached
  features, audit reports, environments, logs, results, and checkpoints during
  code updates or Git setup. They are not transferred by a normal Git pull.
- Build a separate Linux `.venv` on Anunna using uv; never copy the Windows
  environment. Use `--extra encoder` for local CPU commands and
  `--extra encoder-cuda` for NVIDIA CUDA commands. These extras are exclusive.
- The current Anunna `.venv` links to the `skylake_avx512` Python 3.13.1 from
  the A100 nodes. CPU jobs reusing it must request `main --constraint=gen2`
  (Skylake); it raises SIGILL/exit 132 on the Zen3 login/gen3 nodes. Load modules
  inside the allocation. Check `readlink -f .venv/bin/python` and `$EBROOTPYTHON`
  before reusing an environment across CPU architectures; do not retarget or
  recreate the working training environment as a first response to this error.
- Prepare/update the Anunna environment in a compute allocation when the
  dependency configuration changes: `uv sync --locked --extra encoder-cuda`.
  Use the prepared environment in batch jobs without installing packages there.
- Submit substantial computation through Slurm, not directly on login nodes.
  Use VS Code Remote SSH for light inspection/editing, job submission and logs.
- Keep a fixed code version for each active experiment. Pull updates between
  runs; use separate checkouts for concurrent runs needing different versions.
  Preserve configuration, seed, data/cache identity, and code version for each run.
- Review short-run metrics before full training. Bring small reports back for
  analysis and retain important results/checkpoints outside unbacked-up storage.
