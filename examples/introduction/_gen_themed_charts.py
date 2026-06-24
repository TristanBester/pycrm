"""Generate documentation training-curve charts styled to the PyCRM docs site.

Runs the real Letter World Q-learning and counterfactual Q-learning loops, then
renders the smoothed return curves on a dark editorial panel that matches the
docs site palette (magenta / cyan on near-black purple).

Outputs:
    site/public/images/q-learning.png   single Q-learning curve
    site/public/images/cq-learning.png  Q-learning vs counterfactual comparison
"""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from examples.introduction.core.crossproduct import (
    LetterWorldCrossProduct,  # noqa: E402
)
from examples.introduction.core.ground import LetterWorld  # noqa: E402
from examples.introduction.core.label import LetterWorldLabellingFunction  # noqa: E402
from examples.introduction.core.machine import (  # noqa: E402
    LetterWorldCountingRewardMachine,
)

# --- Site palette (approx sRGB of the docs OKLCH tokens) ------------------
PANEL = "#17121f"      # near --bg-elev (dark)
INK = "#cbc7d6"        # near --fg-dim
FAINT = "#8b8699"      # near --fg-faint
GRID = (1, 1, 1, 0.045)
MAGENTA = "#f857a0"    # near --accent
CYAN = "#5fcdde"       # near --accent-2

OUT = Path(__file__).resolve().parents[2] / "site" / "public" / "images"
EPISODES = 5000
SEED = 7


def make_env():
    return LetterWorldCrossProduct(
        ground_env=LetterWorld(),
        crm=LetterWorldCountingRewardMachine(),
        lf=LetterWorldLabellingFunction(),
        max_steps=500,
    )


def train_q_learning(lr=0.1, eps=0.1, gamma=0.99):
    env = make_env()
    q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = []
    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            if np.random.random() < eps or np.all(q[tuple(obs)] == 0):
                action = np.random.randint(env.action_space.n)
            else:
                action = int(np.argmax(q[tuple(obs)]))
            next_obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            done = terminated or truncated
            target = reward if done else reward + gamma * np.max(q[tuple(next_obs)])
            q[tuple(obs)][action] += lr * (target - q[tuple(obs)][action])
            obs = next_obs
        returns.append(ep_return)
    return np.array(returns)


def train_counterfactual(lr=0.01, eps=0.1, gamma=0.99):
    env = make_env()

    def run(counterfactual):
        q = defaultdict(lambda: np.zeros(env.action_space.n))
        returns = []
        for _ in range(EPISODES):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                if np.random.random() < eps or np.all(q[tuple(obs)] == 0):
                    action = np.random.randint(env.action_space.n)
                else:
                    action = int(np.argmax(q[tuple(obs)]))
                next_obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
                if counterfactual:
                    for o, a, o_, r, d, _ in zip(
                        *env.generate_counterfactual_experience(
                            env.to_ground_obs(obs),
                            action,
                            env.to_ground_obs(next_obs),
                        ),
                        strict=True,
                    ):
                        tgt = r if d else r + gamma * np.max(q[tuple(o_)])
                        q[tuple(o)][a] += lr * (tgt - q[tuple(o)][a])
                target = reward if done else reward + gamma * np.max(q[tuple(next_obs)])
                q[tuple(obs)][action] += lr * (target - q[tuple(obs)][action])
                obs = next_obs
            returns.append(ep_return)
        return np.array(returns)

    return run(False), run(True)


def smooth(x, window):
    return np.convolve(x, np.ones(window) / window, mode="valid")


def style_axes(ax):
    fig = ax.figure
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
        ax.spines[side].set_alpha(0.4)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=FAINT, labelsize=11, length=0, pad=8)
    ax.grid(True, axis="y", color=GRID, linestyle="-", linewidth=0.8)
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(FAINT)


def glow_plot(ax, y, color, label=None, lw=2.4):
    (line,) = ax.plot(y, color=color, linewidth=lw, label=label, solid_capstyle="round")
    line.set_path_effects(
        [pe.Stroke(linewidth=lw + 5, foreground=color, alpha=0.12), pe.Normal()]
    )
    return line


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / name,
        dpi=200,
        facecolor=PANEL,
        bbox_inches="tight",
        pad_inches=0.35,
    )
    plt.close(fig)
    print("wrote", OUT / name)


def main():
    np.random.seed(SEED)

    print("training q-learning ...")
    ql = train_q_learning()

    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_axes(ax)
    glow_plot(ax, smooth(ql, 50), MAGENTA)
    ax.set_xlabel("Episode", color=INK, fontsize=12, labelpad=10)
    ax.set_ylabel(
        "Average return  ·  50-episode moving average",
        color=INK,
        fontsize=12,
        labelpad=12,
    )
    save(fig, "q-learning.png")

    print("training counterfactual comparison ...")
    base, cf = train_counterfactual()

    fig, ax = plt.subplots(figsize=(10, 5.4))
    style_axes(ax)
    glow_plot(ax, smooth(base, 100), CYAN, label="Q-learning")
    glow_plot(ax, smooth(cf, 100), MAGENTA, label="Counterfactual Q-learning")
    ax.set_xlabel("Episode", color=INK, fontsize=12, labelpad=10)
    ax.set_ylabel(
        "Average return  ·  100-episode moving average",
        color=INK,
        fontsize=12,
        labelpad=12,
    )
    leg = ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=11.5,
        labelcolor=INK,
        handlelength=1.6,
        borderpad=0.4,
    )
    for text in leg.get_texts():
        text.set_color(INK)
    save(fig, "cq-learning.png")


if __name__ == "__main__":
    main()
