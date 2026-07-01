# LetterWorld TPU Speedup Notebook — Implementation Plan

> **For agentic workers:** Execute task-by-task with superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship a Google Colab notebook that clones the public `experimental/jax` branch, runs the LetterWorld Anakin-vs-SB3 comparison on a **TPU** (Option C: scaled JAX parallelism + raised, matched update ratio), and renders one plot — **wall-clock time (x) vs mean eval return (y)** — with both arms overlaid across 5 seeds, showing the JAX-on-TPU arm reaching high return in less wall-clock than SB3-on-CPU.

**Design spec:** `docs/superpowers/specs/2026-07-01-letterworld-tpu-speedup-notebook-design.md` (read for full rationale).

**Architecture:** Small, backward-compatible parameterization of the two existing demo trainers (expose the update-ratio knob) so the notebook can drive Option C without duplicating logic; a self-contained Colab notebook that installs `jax[tpu]` + repo deps, clones the branch, guards `import stoa` on TPU, runs both arms × 5 seeds, and plots return-vs-wall-clock; a README subsection documenting the config + fairness framing.

**Tech stack:** Python 3.10–3.12, JAX ≥0.6.2 (TPU on Colab), `stoa-env`, `optax`, Stable-Baselines3, matplotlib, nbformat. Reuses `examples/rm/letterworld_anakin/{train_anakin.py, train_sb3.py}`.

## Global Constraints

Every task's requirements implicitly include this section.

- **Honesty & matched fairness are the point.** Config changes from the frozen CPU baseline must be disclosed (notebook markdown + README). `num_envs` scaling is JAX-only (its native execution model); the update ratio is **matched** on both arms (cheap for JAX, expensive for SB3's PyTorch loop — the architectural point); everything else stays identical.
- **No library changes.** Do NOT modify `pycrm/**`, `examples/rm/anakin_puckworld/**`, or `examples/introduction/core/**`. Trainer edits confined to `examples/rm/letterworld_anakin/{train_anakin.py, train_sb3.py}` and MUST be **backward-compatible** (new params default to today's frozen values so `tests/test_letterworld_anakin.py` stays green and the CPU demo is unchanged).
- **CPU backend for all local tests** (`JAX_PLATFORMS=cpu`). The TPU run is the user's Colab step — we cannot run a TPU; do not claim TPU validation.
- **ruff-clean** (line-length 88, google docstrings). Notebook code cells should be import-clean and syntactically valid, but the notebook is not ruff-gated like `.py`.
- **Commit** each task green; append the trailer:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- **Push at the end (Task 3)** so the branch on `github.com/TristanBester/pycrm @ experimental/jax` carries the parameterized trainers + notebook + README before the user opens Colab.

---

## Task 1: Parameterize the two trainers (backward-compatible) + CPU tests

**Files:**
- Modify: `examples/rm/letterworld_anakin/train_anakin.py`
- Modify: `examples/rm/letterworld_anakin/train_sb3.py`
- Test: `tests/test_letterworld_anakin.py` (append)

**Interfaces:**
- `run_anakin_dqn(total_timesteps, seed, log_dir, num_envs=64, env_steps_per_update=4) -> dict` — replaces the hardcoded `// 4` at the `updates_per_rollout` line with `// env_steps_per_update`. Default 4 reproduces current behavior.
- `run_sb3_dqn(total_timesteps, seed, log_dir, env_steps_per_update=4, gradient_steps=1) -> dict` — passes `train_freq=env_steps_per_update, gradient_steps=gradient_steps` into `DQN(...)`. Defaults reproduce current behavior.
- Matched semantics: `env_steps_per_update = E` ⇒ 1 gradient update per E env-steps (ratio 1/E) on both arms.

- [ ] **Step 1 (RED):** Append a smoke test `test_trainers_accept_env_steps_per_update` that (a) calls `run_anakin_dqn(total_timesteps=2048, seed=0, log_dir="/tmp/lw_p", num_envs=8, env_steps_per_update=2)` and asserts the shared schema (backend, equal-length eval lists, finite mean_return); (b) calls `run_sb3_dqn(total_timesteps=3000, seed=0, log_dir="/tmp/lw_p3", env_steps_per_update=2)` and asserts `backend=="sb3_dqn"` and ≥1 eval. Run it: expect FAIL (`TypeError: unexpected keyword argument 'env_steps_per_update'`).
- [ ] **Step 2 (GREEN):** Implement the two signature changes exactly as in Interfaces. In `train_anakin.py`, change `updates_per_rollout = max((num_envs * ROLLOUT) // 4, 1)` → `... // env_steps_per_update ...`. In `train_sb3.py`, thread `train_freq=env_steps_per_update, gradient_steps=gradient_steps` into the `DQN(...)` constructor. Add one-line docstring updates noting the new params.
- [ ] **Step 3:** Run the FULL file: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_letterworld_anakin.py -q` — expect all prior tests + the new one pass (defaults unchanged ⇒ nothing else moves). Confirm `.venv/bin/ruff check examples/rm/letterworld_anakin tests/test_letterworld_anakin.py` clean.
- [ ] **Step 4:** Commit `feat(examples): expose matched update-ratio knob on both letterworld trainers`.

---

## Task 2: Colab notebook

**Files:**
- Create: `notebooks/letterworld_tpu_demo.ipynb`

**Interfaces:** Consumes Task 1's parameterized `run_anakin_dqn` / `run_sb3_dqn`. Produces a valid `nbformat` v4 notebook with the cells below.

- [ ] **Step 1:** Author `notebooks/letterworld_tpu_demo.ipynb` (nbformat v4) with these cells, each preceded by a markdown cell explaining it:
  1. **Runtime/TPU check** — `import jax; print(jax.devices())`; assert a TPU device is present with a clear message if not (and set `USE_TPU`).
  2. **Install** — `pip install -q "jax[tpu]"` for the runtime, then `stoa-env optax stable-baselines3 matplotlib` without downgrading jax; print resolved `jax.__version__` and backend.
  3. **Clone + install repo** — `git clone --branch experimental/jax --depth 1 https://github.com/TristanBester/pycrm.git`; add repo root to `sys.path`; `pip install -e .`.
  4. **⚠️ stoa-on-TPU guard** — `import stoa`; build a small int8 array + a trivial op on the TPU device; on exception, print the failure and set `USE_TPU=False` (fallback to CPU for the JAX arm), matching the Metal failure mode described in the spec.
  5. **Config** — expose `NUM_ENVS`, `ENV_STEPS_PER_UPDATE`, `JAX_BUDGET`, `SB3_BUDGET`, `SEEDS` as editable variables with the spec's starting values (num_envs=512, env_steps_per_update=2, JAX budget ~750k, SB3 budget 150k, 5 seeds); a comment explains the `updates_per_rollout = num_envs*ROLLOUT//E` coupling.
  6. **Run** — loop `for seed in range(SEEDS)`: `run_anakin_dqn(JAX_BUDGET, seed, log_dir, num_envs=NUM_ENVS, env_steps_per_update=ENV_STEPS_PER_UPDATE)` (JAX default backend = TPU when `USE_TPU`) and `run_sb3_dqn(SB3_BUDGET, seed, log_dir, env_steps_per_update=ENV_STEPS_PER_UPDATE)`; collect each dict's `eval_wall`, `mean_return`, `time_to_solve`.
  7. **Plot** — x=wall-clock s, y=mean eval return; both arms overlaid; per arm 5 thin per-seed lines + a bold mean interpolated onto a common wall-clock grid + a ±std band; distinct colors, legend, title; save PNG + display inline. Also print a summary (per-arm mean time_to_solve, solved/5).
  8. **Fairness note (markdown)** — the four framing points from the spec (num_envs JAX-only; matched update ratio; else identical; SB3 on VM CPU).
- [ ] **Step 2:** Validate structurally (no TPU execution): `.venv/bin/python -c "import nbformat; nb=nbformat.read('notebooks/letterworld_tpu_demo.ipynb', as_version=4); nbformat.validate(nb); import ast; [ast.parse(''.join(c.source)) for c in nb.cells if c.cell_type=='code']; print('nbformat valid;', len(nb.cells), 'cells; code cells parse')"`. Expect valid + all code cells parse.
- [ ] **Step 3:** Commit `feat(notebooks): letterworld anakin-vs-sb3 TPU speedup demo notebook`.

---

## Task 3: README TPU subsection + final verification + push

**Files:**
- Modify: `examples/rm/letterworld_anakin/README.md`

- [ ] **Step 1:** Add a "## Running on TPU (Colab)" subsection to the README: what the notebook does; how to open it (Colab → GitHub → `TristanBester/pycrm`, branch `experimental/jax`, `notebooks/letterworld_tpu_demo.ipynb`, select a TPU runtime); the Option-C config (scaled `num_envs`, matched raised update ratio) and the fairness framing (num_envs JAX-only; ratio matched; else identical; SB3 on CPU); and the honest caveat that the TPU result is produced by the user's run and that `import stoa` on TPU is unverified until then (with the CPU fallback).
- [ ] **Step 2 (final verification, report each):**
  - `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_letterworld_anakin.py -q` → all pass.
  - `.venv/bin/ruff check examples/rm/letterworld_anakin` → clean.
  - `.venv/bin/python -c "import nbformat; nbformat.validate(nbformat.read('notebooks/letterworld_tpu_demo.ipynb', as_version=4)); print('nb ok')"` → ok.
- [ ] **Step 3:** Commit `docs(examples): document TPU Colab run + fairness framing`.
- [ ] **Step 4:** Push: `git push origin experimental/jax`. Confirm the remote tip now contains the parameterized trainers, the notebook, and the README subsection.

## Final verification

- [ ] `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_letterworld_anakin.py -q` → all pass.
- [ ] `.venv/bin/ruff check examples/rm/letterworld_anakin` → clean.
- [ ] Notebook is nbformat-valid and all code cells parse.
- [ ] Branch pushed; remote carries trainers + notebook + README.
- [ ] (User step) Notebook runs on Colab TPU and renders the return-vs-wall-clock plot; stoa-on-TPU guard passes or CPU fallback engages.
