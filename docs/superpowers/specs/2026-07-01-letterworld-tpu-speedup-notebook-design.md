# LetterWorld TPU Speedup Notebook — Design Spec

**Status:** Approved (brainstorm 2026-07-01)

## Goal

Produce a Google Colab notebook that clones the public repo's `experimental/jax`
branch, runs the LetterWorld Anakin-vs-SB3 comparison on a **TPU**, and renders a
single deliverable plot — **wall-clock time (x) vs mean eval return (y)** — with
both arms overlaid across 5 seeds, demonstrating that the JAX/Anakin arm on TPU
reaches high return in less wall-clock than a matched Stable-Baselines3 DQN on CPU.

## Context / Motivation

The existing CPU demo (`examples/rm/letterworld_anakin/`) proved the library
`JaxCrossProduct` + JAX-DQN genuinely learn and **solve** LetterWorld, but on CPU
the JAX arm was NOT faster **to solve** (SB3 was ~10× more sample-efficient; JAX
kept only ~7× raw throughput). Two facts shape this follow-on:

- **Metal is dead for this stack.** Empirically verified on 2026-07-01: `jax-metal`
  0.6.2 on M1 Pro fails even a trivial `(x*2).sum()`, and `import stoa` crashes with
  `XlaRuntimeError: UNIMPLEMENTED: default_memory_space is not supported` (stoa runs
  `jnp.array(0, dtype=jnp.int8)` at import). So the accelerator path must be TPU/GPU.
- **TPU is a first-class JAX backend**, far more likely to run stoa than Metal — but
  this is **UNVERIFIED** until run on Colab, and is the make-or-break risk.

The user chose **Option C**: scale the JAX arm's parallelism AND raise the shared
update-ratio lever, to give the TPU a genuine chance to win *time-to-solve* (not just
throughput, which the user explicitly does not want to showcase alone).

## Global Constraints

- **Honesty & matched fairness are the whole point.** Any config change from the
  frozen CPU baseline must be disclosed in-notebook and in the README, and the
  learning comparison must stay apples-to-apples (see Fairness Framing).
- **No library changes.** Do NOT modify `pycrm/**`, `examples/rm/anakin_puckworld/**`,
  or `examples/introduction/core/**`. Trainer changes are confined to
  `examples/rm/letterworld_anakin/{train_anakin.py, train_sb3.py}` and must be
  **backward-compatible** (new params default to today's frozen values, so the
  existing `tests/test_letterworld_anakin.py` and the CPU demo are unaffected).
- **Branch state:** `experimental/jax` is public at
  `github.com/TristanBester/pycrm` (the `counting-reward-machines` URL redirects to
  it) and already carries the complete, fixed demo. Trainer changes here require a
  re-test (CPU) + commit + **push** before the notebook can clone them.
- **Verification split:** trainer changes are CPU-testable by us; the **notebook's
  TPU execution and the `import stoa`-on-TPU guard are validated by the user on
  Colab** — we cannot run a TPU. The spec/plan must not claim TPU validation we
  cannot perform.

## Components

### 1. Trainer parameterization (backward-compatible)

**`run_anakin_dqn(total_timesteps, seed, log_dir, num_envs=64, env_steps_per_update=4)`**
- Replace the hardcoded ratio at `train_anakin.py:339`
  `updates_per_rollout = max((num_envs * ROLLOUT) // 4, 1)`
  with `... // env_steps_per_update ...`.
- `num_envs` already exists. Default `env_steps_per_update=4` reproduces the current
  1-update-per-4-env-steps behavior exactly.

**`run_sb3_dqn(total_timesteps, seed, log_dir, env_steps_per_update=4, gradient_steps=1)`**
- Pass `train_freq=env_steps_per_update, gradient_steps=gradient_steps` into `DQN(...)`.
- Defaults (`4, 1`) reproduce current behavior (SB3 default `train_freq=4`).

**Matched semantics:** for both arms, `env_steps_per_update = E` means **1 gradient
update per E env-steps** (ratio 1/E). Passing the same `E` to both keeps the update
ratio matched; lowering E raises the ratio (cheap for JAX, expensive for SB3's
PyTorch loop). `num_envs` scaling is JAX-only by nature (SB3 is single-env).

### 2. Colab notebook — `notebooks/letterworld_tpu_demo.ipynb`

Cells, run top-to-bottom on a Colab **TPU** runtime:

1. **Runtime/TPU check** — `import jax; print(jax.devices())`; assert a TPU device is
   present (clear message if the runtime isn't TPU).
2. **Install** — `pip install -q "jax[tpu]"` matched to the runtime's libtpu, then the
   repo deps (`stoa-env`, `optax`, `stable-baselines3`) without downgrading jax; print
   resolved versions.
3. **Clone + install repo** —
   `git clone --branch experimental/jax --depth 1 https://github.com/TristanBester/pycrm.git`,
   add repo root to `sys.path`, `pip install -e .`.
4. **⚠️ stoa-on-TPU guard (make-or-break)** — `import stoa`, build a tiny int8 array
   and a small op on the TPU device; on failure, print the failure and set a flag that
   the run cell uses to **fall back to CPU** (still shows the update-ratio effect,
   minus TPU throughput).
5. **Run** — for `seed in range(5)`: call `run_anakin_dqn(...)` (TPU, scaled config)
   and `run_sb3_dqn(...)` (CPU, matched ratio); collect each run dict's
   `(eval_wall, mean_return)` and `time_to_solve`.
6. **Plot** — the deliverable (see §4). Also print a small summary table
   (per-arm mean time_to_solve, solved/5).

### 3. Fairness/README update

Update `examples/rm/letterworld_anakin/README.md` (a new "TPU run" subsection) to
document the Option-C config and the fairness framing, so the notebook's result is
defensible and consistent with the repo.

## Run Config (Option C — starting values, tuned in-notebook)

**Coupling caveat:** JAX `updates_per_rollout = (num_envs × ROLLOUT) // env_steps_per_update`,
so `num_envs` and `env_steps_per_update` **jointly** set the number of gradient updates
per rollout. Large `num_envs` with small `env_steps_per_update` explodes the update
count (e.g. 1024 envs × ROLLOUT 16 // 1 = 16,384 updates/rollout — far too many). The
two knobs must be tuned **together**; below are moderate, likely-stable starting points,
and the notebook exposes both as variables for tuning on the actual TPU.

- **JAX (TPU):** start `num_envs=512`, `env_steps_per_update=2` (updates/rollout =
  512×16//2 = 4096), env-step budget generous (~500k–1M; cheap on TPU, ensures it
  solves). Scale `num_envs` up (1024/2048) to fill the TPU, adjusting
  `env_steps_per_update` upward in step to keep updates/rollout sane.
- **SB3 (CPU):** `env_steps_per_update=2` (same ratio — matched), budget ~150k. Ratio
  kept moderate so 5 CPU seeds finish in reasonable Colab time; this is the knob that
  makes SB3 pay (extra PyTorch backward passes) while JAX pays little.
- **Unchanged / matched:** net arch (64,64), γ=0.99, LR=1e-3, batch=128, buffer=50k,
  target-update semantics, ε 1.0→0.05, eval protocol (K=2000 env-steps, N=20 greedy
  episodes, success = RM `TERMINATED`), 5 seeds.
- **Success criterion for the demo:** the JAX-on-TPU arm reaches near-solved return in
  lower wall-clock than SB3-on-CPU on the overlaid plot. If the moderate starting config
  does not show this, the notebook's tuning variables (num_envs, env_steps_per_update,
  budget) are adjusted on the TPU; the honest fallback is to report what the plot shows.

## Deliverable Plot

- **x = wall-clock seconds, y = mean eval return.** Both arms overlaid; 5 seeds each.
- Per arm: 5 thin per-seed lines + a bold mean interpolated onto a common wall-clock
  grid + a shaded ±std band. Distinct colors per arm; legend; title.
- Expectation (to be confirmed on TPU): the JAX-on-TPU curve reaches high return far
  to the left of the SB3-on-CPU curve.
- Saved to a PNG and displayed inline.

## Fairness Framing (stated in notebook + README)

1. **`num_envs` scaling is JAX-only** — parallel env batching is JAX/Anakin's native
   execution model; SB3 is a single-env off-policy learner by design. Each framework
   runs in its idiomatic mode.
2. **The update ratio is matched** on both arms — identical updates-per-env-step. It is
   cheap for JAX (vectorized/fused) and expensive for SB3 (PyTorch backward passes in a
   Python loop); that asymmetry is the architectural point being demonstrated.
3. **Everything else is identical** (arch, γ, LR, batch, buffer, target-update, ε,
   eval protocol, seeds).
4. **SB3 runs on the TPU VM's CPU** (PyTorch, tiny net — the correct baseline).

## Risks & Fallbacks

- **stoa-on-TPU incompatible** (the Metal failure mode) → guard cell detects it and the
  run falls back to CPU for the JAX arm, with an explicit printed caveat. Result then
  shows the update-ratio effect without TPU throughput.
- **Tiny workload underutilizes the TPU** (64×64 MLP, 3×7 grid) → `num_envs` is a
  notebook variable; scale it up (2048/4096) until the TPU is filled.
- **Host-sync-bound trainer** — `train_anakin.py` does per-iteration host syncs
  (logging/eval), which can cap TPU throughput; large `num_envs` amortizes it. Deeper
  loop-fusion is explicitly OUT OF SCOPE for this notebook.
- **jax[tpu] vs repo dep conflicts** → pin install order (jax[tpu] first), print
  resolved versions, and don't let SB3/optax downgrade jax.

## Testing / Verification

- **CPU (by us, via subagents):** the parameterized trainers keep
  `tests/test_letterworld_anakin.py` green (defaults unchanged); a quick CPU smoke of
  each trainer with a non-default `env_steps_per_update` confirms the new knob threads
  through; ruff clean.
- **TPU (by the user, on Colab):** run the notebook end-to-end; the stoa-on-TPU guard
  passes (or the fallback engages); the plot renders.

## Out of Scope

- Fully fusing the Anakin training loop (removing host syncs).
- GPU (CUDA) variant (TPU is the chosen target; the notebook's CPU fallback covers the
  no-accelerator case).
- Changing the "solved = success_rate ≥ 0.95" definition (a separate open decision).
- Any change to `pycrm/**` or the numpy env.
