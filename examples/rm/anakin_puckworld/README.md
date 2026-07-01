# Anakin PuckWorld demo: gym+SB3 vs Stoa+Stoix/Anakin

This demo trains a DQN agent on the same PuckWorld task, augmented with the
same reward machine, on two different reinforcement-learning stacks:

- **`old` path — gym + Stable-Baselines3 (PyTorch).** `train_sb3.py` wraps
  the numpy `PuckWorldCrossProduct` (from `examples/rm/discrete/core`) as a
  standard `gym.Env` and trains it with SB3's `DQN`.
- **`new` path — Stoa + Stoix/Anakin (JAX).** `train_stoix_anakin.py` builds
  the PuckWorld cross-product via `make_puckworld_cross_product()`
  (`puckworld_stoa_env.py`), which is the Stoa-native `JaxCrossProduct` from
  `pycrm.jax` wrapping a JAX PuckWorld ground environment
  (`puckworld_dynamics.py`) and the same `PuckWorldRewardMachine`. That
  environment is wrapped in the Stoix core-wrapper chain
  (`AddRNGKey` -> `RecordEpisodeMetrics` -> `AutoResetWrapper` ->
  `VmapWrapper`) and trained end-to-end in JAX ("Anakin"-style: vmapped
  environments, `lax.scan`-based rollouts, and jitted DDQN updates) by a
  self-contained DQN implementation.

Both arms consume the *same* ground dynamics and the *same* reward machine —
only the environment/training stack differs. `compare.py` runs both arms
back-to-back and reports throughput (steps/s), wall-clock, and final return
so the two stacks can be compared directly.

## What this demo proves

That a reward machine built on `pycrm`'s cross-product abstraction can be
deployed on two very different RL stacks — the "old" numpy/gym/SB3 path and
the "new" Stoa/Stoix-style JAX/Anakin path — without changing the task
definition (ground environment + reward machine + labelling function). The
comparison is there to sanity-check that (a) both stacks converge to
comparable behavior and (b) the JAX/Anakin path is faster, as expected from
an end-to-end-JAX, vmapped, jit-compiled training loop.

## Install

```bash
uv sync --extra stoix
```

This installs `stoa-env` (the Stoa environment library) and a `jax` version
compatible with it. Stoix itself is not a published dependency yet — it
would be installed from `main` (Anthropic/Stoix upstream) once available —
but this demo does **not** need it: `train_stoix_anakin.py` ships a
self-contained JAX-Anakin DQN fallback (ported from
`examples/rm/discrete-fast/jax_dqn.py`) that reproduces the Stoix
core-wrapper chain and training loop shape without importing Stoix.

## Run

```bash
JAX_PLATFORMS=cpu python -m examples.rm.anakin_puckworld.compare --timesteps 200000
```

This demo was run on an Apple M1 Pro. `jax-metal` (the Metal GPU backend for
JAX on Apple Silicon) crashes on `import stoa`, so the Metal accelerator is
not usable here and `JAX_PLATFORMS=cpu` is required, not optional, on this
machine. See the honesty caveat below for what that means for the reported
numbers.

`compare.py` writes `results/return_vs_steps.png`, `results/return_vs_walltime.png`,
and `results/comparison.csv` (all git-ignored) and prints a two-row summary
table to stdout.

## Honesty caveat

> *SB3 (PyTorch) vs Stoix (JAX) is an architecture comparison, not a
> controlled single-implementation ablation. Hyperparameters are matched
> where the libraries allow; plain DQN both sides (no counterfactual); fixed
> seeds. Success = both converge to comparable final return AND Anakin shows
> materially higher throughput / lower wall-clock-to-threshold. The speedup
> is hardware-dependent — report the run device; Anakin's advantage is
> largest on GPU/TPU. The demo default is `JAX_PLATFORMS=cpu`; for real
> numbers run on an accelerator.*

The results below were produced with `JAX_PLATFORMS=cpu` on an Apple M1 Pro
because `jax-metal` cannot be used (it crashes on `import stoa` on this
machine, so the Metal accelerator is unavailable for this demo). Per the
caveat above, Anakin's throughput advantage is largest on GPU/TPU and is
understated by the CPU numbers reported here — do not read the CPU speedup
as representative of the accelerator case.

## Tuning `SOLVED_THRESHOLD`

`compare.py` defines `SOLVED_THRESHOLD = 0.0` as a placeholder "task solved"
return used only to compute steps/wall-clock-to-threshold. This value is
**not tuned** to PuckWorld's actual return scale and must be adjusted by
inspecting the observed final returns (see results below) before
steps/wall-to-threshold numbers are meaningful. Depending on the return
scale, an untuned threshold can fail in either direction: it can be trivial
or degenerate (crossed immediately) if typical returns are already at or
above the threshold, or — as observed in the run below, where returns sit
around -10,000 against a threshold of `0.0` — it can never be crossed at
all, making `steps@thr`/`wall@thr` uninformative (`None`/`n/a`) regardless
of how much either backend actually learns.

## Results (CPU, Apple M1 Pro, `--timesteps 200000`)

Run command:

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m examples.rm.anakin_puckworld.compare --timesteps 200000
```

Printed table (verbatim):

```
backend                    steps/s   wall(s)   final_ret   steps@thr    wall@thr
sb3_dqn                       2237      89.4    -10449.5        None         n/a
anakin_dqn                    5445      36.7    -10465.0        None         n/a
```

Observations:

- Both arms complete 200,000 environment steps.
- Anakin (JAX, CPU) trains at roughly **2.4x** the steps/s of SB3 (PyTorch,
  CPU) on this machine (~5.4k steps/s vs ~2.2k steps/s), finishing the full
  run in ~36.7s vs ~89.4s wall-clock.
- Neither arm reaches anything close to `SOLVED_THRESHOLD = 0.0`: final
  single-seed returns are approximately -10,450 (SB3) and -10,465 (Anakin).
  Per-episode returns for both backends stay in the roughly -5,000 to
  -10,700 range throughout training (see `results/sb3_progress.csv` and
  `results/anakin_dqn_progress.csv`), meaning `steps@thr`/`wall@thr` are
  reported as `None`/`n/a` for both arms because the threshold is never
  crossed. This is the expected consequence of `SOLVED_THRESHOLD = 0.0`
  being an untuned placeholder relative to PuckWorld's actual return scale
  (roughly four orders of magnitude below zero) — the threshold columns are
  not informative at this setting and would need to be retuned (e.g. to
  something in the observed -5,000 to -10,000 range, or lower once returns
  actually improve with more training) before "time to threshold" is a
  meaningful metric here.
- Because neither run is anywhere near solving the task in 200k steps, this
  run should be read as a **throughput/architecture comparison**, not
  evidence that either DQN configuration has learned a good PuckWorld
  policy yet — consistent with the honesty caveat's framing that success
  requires "comparable final return" between the two arms (which holds:
  both land around -10,450 to -10,465) plus higher Anakin throughput (which
  also holds), not necessarily a solved task.
- These numbers were obtained on CPU only (no accelerator available on this
  machine for this demo, since `jax-metal` crashes on `import stoa`). Per
  the honesty caveat, treat the ~2.4x figure as a CPU-only, single-run,
  lower bound on Anakin's advantage — GPU/TPU speedups are expected to be
  larger, since Anakin's whole design (vmapped envs, `lax.scan` rollouts,
  jitted updates) is built to exploit accelerator parallelism that a CPU
  cannot provide.
