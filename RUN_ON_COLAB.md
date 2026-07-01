# Run the TPU speed-up demo on Google Colab

One notebook trains the JAX/Anakin DQN and the Stable-Baselines3 DQN on LetterWorld
and plots **mean return vs wall-clock** for both arms. To run it:

1. **Open the notebook** (one click):
   <https://colab.research.google.com/github/TristanBester/pycrm/blob/experimental/jax/notebooks/letterworld_tpu_demo.ipynb>
   *(or in Colab: File → Open notebook → GitHub → `TristanBester/pycrm` → branch `experimental/jax` → `notebooks/letterworld_tpu_demo.ipynb`)*

2. **Pick a TPU:** Runtime → Change runtime type → **TPU** → Save.

3. **Run everything:** Runtime → **Run all**. (The first run installs dependencies — a few minutes.)

4. **Check the guard cell** ("stoa-on-TPU guard"). If it prints OK, the run continues on the TPU. If it errors, do what it says: add a cell at the very top with
   `import os; os.environ["JAX_PLATFORMS"] = "cpu"`, then Runtime → **Restart runtime**, and Run all again (CPU fallback).

5. **Get the result:** the final cells print a per-arm summary and display the plot
   (x = wall-clock seconds, y = mean return, both arms overlaid). It's also saved to
   `/content/letterworld_tpu_return_vs_time.png` — download it from the Files pane (📁 on the left).

**If the JAX curve isn't clearly faster:** LetterWorld is tiny, so bump the parallelism —
in the config cell set `NUM_ENVS = 1024` (or `2048`) and Run all again.

More detail and the fairness notes: [`examples/rm/letterworld_anakin/README.md`](examples/rm/letterworld_anakin/README.md).
