# LetterWorld: Anakin-JAX vs. SB3 convergence demo

A small, disposable **correctness** demo. It checks that the library's JAX
reward-machine cross-product (`pycrm.jax.JaxCrossProduct`) plus a from-scratch
JAX single-DQN trainer genuinely **learn and SOLVE** a counting-reward-machine
task (LetterWorld), and compares that JAX arm head-to-head against a matched
Stable-Baselines3 DQN baseline running on the read-only numpy cross-product.

## 1. What it validates

- **The JAX RM stack learns.** `JaxCrossProduct` composes the pure-JAX
  LetterWorld ground dynamics with the compiled `LetterWorldCountingRewardMachine`
  and produces a well-shaped observation/reward stream that a plain DQN can
  optimize end-to-end inside JAX (vmapped envs + `lax.scan` rollouts + jitted
  updates). "Solved" means a greedy eval reaches the reward-machine terminal
  (`StepType.TERMINATED`) within `max_steps` on at least 95% of episodes.
- **Parity of task, not just of code.** The SB3 arm learns on the original,
  read-only numpy `LetterWorldCrossProduct` from `examples/introduction/core`,
  so the two arms are solving the *same* reward-machine task through two
  different library implementations.

## 2. Fairness protocol ("matched except backend")

Both arms use a **single (vanilla) DQN** — no double-DQN, no dueling, no PER —
with an **identical, frozen** hyperparameter set (frozen in the Task-3 spike;
changing them for a nicer number is forbidden):

| Hyperparameter        | Value            |
| --------------------- | ---------------- |
| `HIDDEN`              | `(64, 64)`       |
| `GAMMA`               | `0.99`           |
| `LR`                  | `1e-3`           |
| `BUFFER`              | `50_000`         |
| `BATCH`               | `128`            |
| `TARGET_UPDATE`       | `500`            |
| `LEARNING_STARTS`     | `1_000`          |
| `EXPLORATION_FRACTION`| `0.2`            |
| epsilon schedule      | `1.0 -> 0.05`    |
| `MAX_STEPS`           | `100`            |

Other matched details:

- **Correct truncation bootstrapping.** The JAX trainer bootstraps on **true
  termination only** (`StepType.TERMINATED`), never on time-limit truncation, so
  a `max_steps` cut-off does not zero out the bootstrap value. SB3 handles this
  the same way for its `TimeLimit`/`Monitor`-wrapped env.
- **Identical greedy eval.** Every `K = 2000` env-steps, run `N = 20` greedy
  episodes; success = reaching the RM terminal within `max_steps`. Same K and N
  on both arms.
- **5 seeds** (0-4) per arm.
- **Target-update cadence is not byte-identical.** Because the JAX arm hard-copies its target network once per rollout (`num_envs=64` x `ROLLOUT=16` = 1024 env-steps, and `max(500 // 1024, 1) = 1` rollout), it refreshes the target roughly every 1024 env-steps versus SB3's 500 (`target_update_interval=500`); this rollout-granularity/integer-floor asymmetry slightly disfavors the JAX arm and does not change the reported conclusion.

### Fairness nuance: the observation encodings are *not* byte-identical

The two library cross-product implementations emit slightly different
observation vectors — an inherent difference between the JAX and numpy
cross-products, not a knob we tuned:

- **JAX arm (`JaxCrossProduct`): 7-dim** — `3` ground features
  `[symbol_seen, row, col]` + `one_hot(u)` over the **3** accessible machine
  states + `1` counter value.
- **SB3 arm (numpy `LetterWorldCrossProduct`): 5-dim** —
  `[symbol_seen, row, col, u, c]` with a **scalar** machine state `u` and a
  **scalar** counter `c` (no one-hot).

So "matched except backend" refers to the **learning algorithm, the
hyperparameters, and the eval protocol** — *not* a byte-identical observation
vector. State this plainly when citing the comparison: the JAX arm sees a
one-hot machine-state encoding while SB3 sees a scalar one. (The one-hot layout
is generally easier for an MLP; the numbers below do not favor the JAX arm, so
this nuance does not flatter it.)

## 3. How to run

```bash
JAX_PLATFORMS=cpu python -m examples.rm.letterworld_anakin.compare \
    --timesteps 150000 --seeds 5
```

This runs 5 seeds x 2 arms x 150k env-steps. The JAX arm is seconds per seed;
the SB3 (PyTorch) arm is the slow part — expect several minutes total on CPU.
Outputs (learning curves, `comparison.csv`, PNG plots) are written under
`results/letterworld/` (git-ignored).

## 4. Actual observed numbers (5 seeds, 150k steps, CPU)

Verbatim from the harness table:

```
backend         solved/seeds     t2solve mean±std(s)      steps2solve mean±std
anakin_dqn               4/5                 4.5±1.1               97024±33468
sb3_dqn                  5/5                 2.6±0.8                 9600±2939
```

`t2solve` = wall-clock (s) to the first eval that reaches >=0.95 success;
`steps2solve` = env-steps to that eval. Means/stds are over the **solved** seeds.

### Honest read of the result

**This run did NOT match the plan's hope of "both arms solve 5/5 and the JAX
arm is faster in wall-clock-to-solve."** What actually happened, on CPU:

- **The JAX/Anakin arm solved 4/5 seeds, not 5/5.** Four seeds clearly reached
  100% greedy success. The fifth (seed 3) plateaued at exactly 19/20 = 0.95 in
  float32 (`0.94999...`, just *below* the >=0.95 bar) and then destabilized
  (final greedy success collapsed to ~0.30). So the JAX arm demonstrably
  *learns and solves* LetterWorld — but under these frozen HPs it is not a
  reliable 5/5 and one seed was unstable.
- **The SB3 arm solved 5/5 seeds** and was **more sample-efficient and faster
  in wall-clock-to-solve**: it crossed the 0.95 bar at ~9,600 env-steps
  (vs ~97,024 for the JAX arm) and at ~2.6 s mean wall-clock (vs ~4.5 s).
  Because SB3 converges in an order of magnitude fewer env-steps, it reaches the
  solve threshold sooner in wall-clock despite its slower per-step throughput.
- **Raw throughput still favors JAX (as expected), but that is a different
  metric.** Finishing the full 150k-step budget took the JAX arm ~6.4 s/seed
  vs ~47 s/seed for SB3 (~7x faster per step). Time-to-*solve*, however, is
  dominated by *sample efficiency*, and here SB3 wins — so on CPU the JAX arm
  is **not** faster in wall-clock-to-solve.

**Bottom line:** the demo's core claim — *the library's `JaxCrossProduct` +
JAX DQN genuinely learn and solve a reward-machine task* — is confirmed (4/5
seeds solved, best-case 100% greedy success). The secondary hope that the JAX
arm would also win wall-clock-to-solve on CPU did **not** hold: SB3 solved 5/5
and reached the bar sooner because it needed ~10x fewer env-steps.

## 5. Caveats

> Run on CPU (jax-metal breaks `import stoa`); the JAX throughput advantage —
> and thus wall-clock-to-solve — is largest on GPU/TPU. Single/other seeds may
> vary; success = reaching the reward-machine terminal within max_steps. This is
> a disposable correctness demo.

The wall-clock-to-solve comparison in particular is CPU-specific: on GPU/TPU the
JAX arm's per-step throughput advantage grows by orders of magnitude, which can
flip the wall-clock-to-solve result even though sample efficiency (env-steps to
solve) is a backend-independent property that would still favor SB3 here.

## Running on TPU (Colab)

The CPU numbers above show sample efficiency favors SB3; the point of a TPU run
is the *other* axis — raw throughput. The companion notebook
`notebooks/letterworld_tpu_demo.ipynb` runs this comparison on a Colab TPU so
you can see the throughput asymmetry directly.

**What the notebook does.** It shallow-clones the `experimental/jax` branch,
installs `jax[tpu]` plus the demo deps, then runs the same Anakin-vs-SB3
LetterWorld comparison across **5 seeds** on a TPU. It plots **mean eval return
(y) vs wall-clock seconds (x)** with both arms overlaid — per-seed curves plus a
mean±std band — so the JAX curve reaching high return far to the left (less
wall-clock) is the deliverable.

**How to open it.** In Colab: *File → Open notebook → GitHub*, enter
`TristanBester/pycrm`, select branch **`experimental/jax`**, and open
`notebooks/letterworld_tpu_demo.ipynb`. Then set the accelerator via
*Runtime → Change runtime type → **TPU*** before running from the top.

**The Option-C config.** The JAX arm scales `num_envs` (its native vmapped
parallelism) to fill the TPU, and **both** arms use a matched, raised update
ratio via `env_steps_per_update`. The notebook's default knobs — all editable in
its config cell — are:

- `NUM_ENVS = 512` (JAX-only; try 1024 / 2048 to fill the TPU),
- `ENV_STEPS_PER_UPDATE = 2` (the matched update ratio, applied to both arms),
- JAX budget `750_000` env-steps, SB3 budget `150_000` env-steps,
- `SEEDS = 5`.

**Fairness framing (same four points as the notebook).**

1. **`num_envs` scaling is JAX-only** — running hundreds/thousands of vmapped
   environments per rollout is JAX's native execution model, not an edge handed
   to it; SB3 has no equivalent knob here.
2. **The update ratio is matched on both arms.** `env_steps_per_update` (and the
   resulting updates/rollout) is applied identically to Anakin and SB3. The same
   ratio is **cheap** for a jitted JAX kernel and **expensive** for SB3's PyTorch
   training loop — that asymmetry, at equal work, is the architectural point.
3. **Everything else is identical:** network architecture, discount γ, learning
   rate, batch size, replay-buffer size, target-update interval, the ε-greedy
   schedule, and the eval protocol (every `K = 2000` env-steps over `N = 20`
   greedy episodes, "solved" at ≥ 0.95 success).
4. **SB3 runs on the TPU VM's CPU.** Stable-Baselines3 uses PyTorch, which is
   **not** using the TPU here — it runs on the host CPU of the same Colab VM.

**Honest caveats.**

- **The TPU result is produced by *your own* Colab run** — it was **not** run or
  validated by the authors. Only the CPU numbers in §4 are author-observed.
- **`import stoa` on TPU is unverified** until you run it. The notebook includes a
  make-or-break guard that imports `stoa` and forces a trivial op onto the TPU
  device before the full run. If it fails (mirroring the confirmed `jax-metal`
  incompatibility that breaks `import stoa`), the documented fallback is a **CPU
  restart**: add a top cell setting `os.environ["JAX_PLATFORMS"] = "cpu"` before
  any `import jax`, *Runtime → Restart runtime*, then re-run — JAX binds its
  backend once per kernel and cannot be hot-swapped mid-session.
- **LetterWorld is tiny**, so the TPU only pays off with enough parallelism. You
  may need to scale `NUM_ENVS` up (1024 / 2048) to see the throughput advantage;
  at small `num_envs` the TPU can be under-fed and unimpressive.
- **The `K = 2000` eval and 500-env-step target-update intervals are exact at the
  frozen defaults but coarsen as you scale `NUM_ENVS`.** A JAX rollout spans
  `num_envs × ROLLOUT` env-steps, so at e.g. `NUM_ENVS = 512` a rollout is 8192
  env-steps and the JAX arm's target-sync and eval each fire once per rollout
  (≈ every 8192 steps) rather than every 500 / 2000. This is a batched-execution
  granularity artifact — if anything it slightly *disfavors* the JAX arm (fewer
  target refreshes, coarser eval curve), not a tilt in its favor.
