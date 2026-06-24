import Link from "next/link";
import { SectionKicker } from "@/components/section-kicker";
import { InstallCommand } from "@/components/install-command";
import { FeatureBlock } from "@/components/feature-block";
import { Receipts } from "@/components/receipts";
import { CodeWindow, type CodeTab } from "@/components/code-window";
import { highlight } from "@/lib/highlight";

const stats: [string, string, string][] = [
  ["2", "machine families", "Reward Machines and Counting Reward Machines share one workflow."],
  ["4", "learning surfaces", "Tabular, SB3, SBX, and JAX examples cover the main research loops."],
  ["14", "docs chapters", "A compact path from first install to counterfactual Q-learning."],
  ["1", "product MDP", "Environment dynamics and automaton memory become one Gymnasium task."],
];

const pipeline: {
  id: string;
  kicker: string;
  title: string;
  text: string;
  specs: [string, string][];
}[] = [
  {
    id: "01",
    kicker: "label",
    title: "Label the transitions.",
    text: "Map raw transitions to a symbolic vocabulary: A, B, C, delivery, pickup, or any event your task logic needs.",
    specs: [
      ["input", "raw transition"],
      ["output", "event label"],
      ["vocabulary", "A · B · C · ∅"],
    ],
  },
  {
    id: "02",
    kicker: "specify",
    title: "Specify the reward logic.",
    text: "Write the reward logic as an RM or CRM, keeping task progress separate from environment dynamics.",
    specs: [
      ["object", "RM · CRM"],
      ["state", "u ∈ U"],
      ["counter", "c ∈ ℤⁿ"],
    ],
  },
  {
    id: "03",
    kicker: "wrap",
    title: "Wrap the cross-product.",
    text: "Build the cross-product so the learner sees a Markov state with machine state and counters attached.",
    specs: [
      ["wrapper", "CrossProduct"],
      ["obs", "[ground, u, c]"],
      ["api", "Gymnasium"],
    ],
  },
  {
    id: "04",
    kicker: "learn",
    title: "Learn with structure.",
    text: "Train standard or counterfactual agents that exploit the automaton structure instead of rediscovering it.",
    specs: [
      ["agents", "tabular · SB3 · SBX · JAX"],
      ["update", "counterfactual"],
      ["target", "all machine states"],
    ],
  },
];

const capabilities: [string, string, string][] = [
  ["rm", "Reward Machines", "Finite-state reward specifications with symbolic transitions."],
  ["crm", "Counting Reward Machines", "Register counters for tasks plain automata cannot express."],
  ["env", "Cross-Product Envs", "Ground state × machine state as one Gymnasium MDP."],
  ["label", "Labelling Functions", "Map transitions to the symbolic events your logic reads."],
  ["tabular", "Tabular Q-Learning", "Reference agents for small, fully inspectable problems."],
  ["sb3", "Stable-Baselines3", "Drop CRM tasks straight into the SB3 training loop."],
  ["jax", "SBX · JAX", "Vectorised, JIT-compiled training on accelerators."],
  ["cf", "Counterfactual Updates", "Learn from every valid machine state, every step."],
];

const compareRows: [string, string, string][] = [
  ["Reward logic", "Inspectable automaton", "Tangled inside step()"],
  ["Task memory", "Explicit state + counters", "Implicit or absent"],
  ["Non-Markovian tasks", "First-class", "Reward hacking"],
  ["Sample efficiency", "Counterfactual reuse", "One update per step"],
  ["Swap the learner", "Tabular → JAX, same task", "Rewrite every time"],
];

const docs: [string, string, string][] = [
  ["Get Started", "Install PyCRM, read the mental model, and run the first Letter World step.", "/docs"],
  ["Letter World", "Walk through the full environment, CRM, wrapper, Q-learning, and counterfactual update.", "/worked-examples/letter-env/setup"],
  ["Core Concepts", "Understand labels, automata, cross-products, and the agent integrations.", "/core-concepts/labelling-functions"],
];

const snippets: { label: string; lang: string; code: string }[] = [
  {
    label: "define",
    lang: "python",
    code: `from pyrewardmachines import CountingRewardMachine


class Deliver(CountingRewardMachine):
    """Reward +1 once every package is picked up, then dropped."""

    def transitions(self, u, props):
        # (machine state, event) -> (next state, counter update)
        if u == 0 and "pickup" in props:
            return 0, ("+1",)          # count each pickup
        if u == 0 and "dropoff" in props:
            return 1, ("-z",)          # clear the counter, advance
        return u, ("0",)

    def reward(self, u, c):
        return 1.0 if u == 1 and c[0] == 0 else 0.0
`,
  },
  {
    label: "wrap",
    lang: "python",
    code: `import gymnasium as gym
from pyrewardmachines import CrossProduct

ground = gym.make("LetterWorld-v0")
machine = Deliver()

# One Gymnasium task: ground state x machine state x counters
task = CrossProduct(ground, machine, labelling_fn=letter_labels)

obs, info = task.reset(seed=0)
obs, reward, done, truncated, info = task.step(action)
`,
  },
  {
    label: "train",
    lang: "python",
    code: `from pyrewardmachines.agents import CounterfactualQLearning

agent = CounterfactualQLearning(task)

# Each transition updates every reachable machine state at once
agent.learn(total_episodes=2_000)

print(agent.evaluate(episodes=100))   # success rate, avg return
agent.save("deliver.npz")
`,
  },
];

export default async function Home() {
  const codeTabs: CodeTab[] = await Promise.all(
    snippets.map(async (s) => ({
      label: s.label,
      lang: s.lang,
      raw: s.code,
      html: await highlight(s.code, s.lang),
    })),
  );

  return (
    <main id="main">
      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="relative overflow-clip border-b hairline">
        <span className="grid-bg pointer-events-none absolute inset-0 opacity-50" aria-hidden="true" />
        <div className="relative mx-auto grid min-h-[calc(78vh-4rem)] max-w-[1280px] items-center gap-12 px-5 py-16 sm:px-8 sm:py-20 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
          <div>
            <p className="mono-label text-[var(--accent)]">
              pycrm // <span className="live">reward logic as code</span>
            </p>
            <h1 className="display mt-7 max-w-[15ch] text-[clamp(2.5rem,1.6rem+4vw,5.6rem)] font-medium leading-[0.94]">
              Counting rewards,{" "}
              <span className="italic text-[var(--fg-dim)]">without hiding</span>{" "}
              <span className="text-gradient">the rules.</span>
            </h1>
            <p className="mt-8 max-w-[56ch] text-[clamp(1rem,0.9rem+0.35vw,1.18rem)] leading-relaxed text-[var(--fg-dim)]">
              Reward Machines, Counting Reward Machines, cross-product
              environments, and counterfactual reinforcement learning — one
              inspectable pipeline from a symbol you can observe to an agent that
              exploits the structure.
            </p>
            <div className="mt-9 flex flex-wrap items-center gap-3">
              <Link href="/docs" className="button-link">
                Read the docs
              </Link>
              <Link
                href="/worked-examples/letter-env/ground-environment"
                className="button-link"
              >
                Letter World
              </Link>
            </div>
            <InstallCommand className="mt-7 max-w-[30rem]" />
            <dl className="mt-10 grid max-w-[34rem] grid-cols-2 gap-x-8 gap-y-6 sm:grid-cols-4">
              {stats.map(([value, label]) => (
                <div key={label}>
                  <dt className="display text-[1.9rem] leading-none text-[var(--fg)]">{value}</dt>
                  <dd className="mt-2 mono-label text-[var(--fg-faint)]">{label}</dd>
                </div>
              ))}
            </dl>
          </div>

          <div className="relative">
            <img
              src="/favicon.svg"
              alt=""
              aria-hidden="true"
              className="drift pointer-events-none absolute -right-6 -top-16 hidden w-[min(22rem,38vw)] opacity-[0.13] blur-[1px] lg:block"
            />
            <div className="terminal reveal edge-card gradient-edge relative border hairline shadow-[0_30px_90px_-40px_var(--accent-iris)]">
              <div className="flex items-center gap-1.5 border-b hairline px-3 py-2">
                <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent)]" />
                <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent-iris)]" />
                <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent-2)]" />
                <span className="ml-auto mono-label text-[var(--fg-faint)]">crm trace</span>
              </div>
              <div className="space-y-4 p-5">
                <p className="text-[var(--fg-faint)]">$ pip install pyrewardmachines</p>
                <p>
                  <span className="text-[var(--accent)]">label</span> transition -&gt; {"{A, B, C}"}
                </p>
                <p>
                  <span className="text-[var(--accent-iris)]">machine</span> state -&gt; u=2, c=(1,)
                </p>
                <p>
                  <span className="text-[var(--accent-2)]">product</span> obs -&gt; [ground, u, c]
                </p>
                <p>
                  <span className="text-[var(--fg)]">counterfactual update</span> -&gt; all valid machine states
                  <span className="caret ml-1" />
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Why / receipts ───────────────────────────────────── */}
      <section className="border-b hairline py-16 sm:py-24">
        <div className="mx-auto max-w-[1280px] px-5 sm:px-8">
          <SectionKicker index="01" title="why it changed" meta="visible artifact" />
          <div className="mt-10 grid items-start gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div>
              <h2 className="display max-w-[18ch] text-[clamp(1.9rem,1.4rem+1.6vw,2.9rem)] font-medium leading-[1.04]">
                The reward function is now a{" "}
                <span className="text-gradient">visible artifact.</span>
              </h2>
              <p className="mt-6 max-w-[52ch] text-[1rem] leading-relaxed text-[var(--fg-dim)]">
                The old docs explained the pieces. This site turns the pieces
                into a route: start with the signal you can observe, encode the
                objective, bind it to the environment, then train with structure
                instead of treating reward as a scalar afterthought.
              </p>
              <div className="mt-9 grid grid-cols-2 gap-px border hairline bg-[var(--rule)]">
                {stats.map(([value, label, text]) => (
                  <div key={label} className="edge-card bg-[var(--bg)] p-5">
                    <div className="display text-[1.9rem] leading-none text-[var(--fg)]">{value}</div>
                    <div className="mt-3 mono-label text-[var(--accent)]">{label}</div>
                    <p className="mt-3 text-[13px] leading-relaxed text-[var(--fg-dim)]">{text}</p>
                  </div>
                ))}
              </div>
            </div>
            <Receipts
              caption="receipts · counterfactual learning"
              rightLabel="letter world"
              footnote="illustrative · tabular q-learning · single-seed trace"
              rows={[
                {
                  label: "sample efficiency",
                  metric: "≈3× fewer",
                  text: "Episodes to converge versus a scalar-reward baseline, by reusing the machine structure the task already encodes.",
                },
                {
                  label: "updates per step",
                  metric: "all states",
                  text: "Every reachable machine state is updated on each environment step — not only the one you happened to visit.",
                },
                {
                  label: "reward leakage",
                  metric: "0",
                  text: "Task progress lives in the automaton, so dynamics and objective never blur into one opaque scalar.",
                },
              ]}
            />
          </div>
        </div>
      </section>

      {/* ── Workflow / feature blocks ────────────────────────── */}
      <section className="relative overflow-clip border-b hairline py-16 sm:py-24">
        <span className="grid-bg pointer-events-none absolute inset-0 opacity-35" aria-hidden="true" />
        <div className="relative mx-auto max-w-[1280px] px-5 sm:px-8">
          <SectionKicker index="02" title="the workflow" meta="four moves" />
          <h2 className="display mt-10 max-w-[20ch] text-[clamp(1.9rem,1.4rem+1.6vw,2.9rem)] font-medium leading-[1.04]">
            Four moves from world dynamics to{" "}
            <span className="text-gradient">structured learning.</span>
          </h2>
          <p className="mt-6 max-w-[58ch] text-[var(--fg-dim)]">
            Each chapter keeps the same contract: what object you are building,
            why it exists, where it plugs in, and how it changes the learner.
          </p>
          <div className="mt-10">
            {pipeline.map((item) => (
              <FeatureBlock
                key={item.id}
                index={item.id}
                kicker={item.kicker}
                title={item.title}
                specs={item.specs}
              >
                {item.text}
              </FeatureBlock>
            ))}
          </div>
        </div>
      </section>

      {/* ── What's in the box ────────────────────────────────── */}
      <section className="border-b hairline py-16 sm:py-24">
        <div className="mx-auto max-w-[1280px] px-5 sm:px-8">
          <SectionKicker index="03" title="what's in the box" meta="batteries included" />
          <h2 className="display mt-10 max-w-[22ch] text-[clamp(1.9rem,1.4rem+1.6vw,2.9rem)] font-medium leading-[1.04]">
            Everything from a label to a{" "}
            <span className="text-gradient">trained agent.</span>
          </h2>
          <div className="mt-10 grid gap-px border hairline bg-[var(--rule)] sm:grid-cols-2 lg:grid-cols-4">
            {capabilities.map(([tag, title, text]) => (
              <div key={tag} className="edge-card group bg-[var(--bg)] p-5">
                <div className="mono-label text-[var(--accent)]">{tag}</div>
                <h3 className="mt-5 display text-[1.2rem] leading-tight text-[var(--fg)]">{title}</h3>
                <p className="mt-3 text-[0.88rem] leading-relaxed text-[var(--fg-dim)]">{text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Code showcase ────────────────────────────────────── */}
      <section className="relative overflow-clip border-b hairline py-16 sm:py-24">
        <span className="grid-bg pointer-events-none absolute inset-0 opacity-35" aria-hidden="true" />
        <div className="relative mx-auto max-w-[1280px] px-5 sm:px-8">
          <SectionKicker index="04" title="see it in code" meta="python · gymnasium" />
          <div className="mt-10 grid items-start gap-12 lg:grid-cols-[minmax(0,0.78fr)_minmax(0,1.22fr)]">
            <div className="lg:pt-6">
              <h2 className="display max-w-[16ch] text-[clamp(1.9rem,1.4rem+1.6vw,2.7rem)] font-medium leading-[1.04]">
                Define it. Wrap it.{" "}
                <span className="text-gradient">Train it.</span>
              </h2>
              <p className="mt-6 max-w-[44ch] text-[var(--fg-dim)]">
                A Counting Reward Machine is just a class. The cross-product makes
                it a Gymnasium task. Any agent — tabular to JAX — trains against
                the same interface.
              </p>
              <Link href="/quickstart" className="button-link mt-8">
                Full quickstart
              </Link>
            </div>
            <CodeWindow tabs={codeTabs} />
          </div>
        </div>
      </section>

      {/* ── Compare ──────────────────────────────────────────── */}
      <section className="border-b hairline py-16 sm:py-24">
        <div className="mx-auto max-w-[1280px] px-5 sm:px-8">
          <SectionKicker index="05" title="structure vs shaping" meta="the difference" />
          <h2 className="display mt-10 max-w-[20ch] text-[clamp(1.9rem,1.4rem+1.6vw,2.9rem)] font-medium leading-[1.04]">
            Stop hiding the objective in a{" "}
            <span className="text-gradient">scalar.</span>
          </h2>
          <div className="mt-10 overflow-x-auto">
            <table className="compare-table w-full min-w-[40rem] border-collapse">
              <thead>
                <tr>
                  <th className="w-1/3" />
                  <th className="text-[var(--fg)]">
                    <span className="inline-flex items-center gap-2">
                      <img src="/favicon.svg" alt="" className="h-4 w-4 rounded-[4px]" />
                      PyCRM
                    </span>
                  </th>
                  <th className="text-[var(--fg-faint)]">Hand-rolled shaping</th>
                </tr>
              </thead>
              <tbody>
                {compareRows.map(([dim, ours, theirs]) => (
                  <tr key={dim}>
                    <td className="mono-label text-[var(--fg-faint)]">{dim}</td>
                    <td className="text-[var(--fg)]">
                      <span className="mr-2 text-[var(--accent-3)]">✓</span>
                      {ours}
                    </td>
                    <td className="text-[var(--fg-dim)]">
                      <span className="mr-2 text-[var(--fg-faint)]">·</span>
                      {theirs}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* ── Docs tracks ──────────────────────────────────────── */}
      <section className="border-b hairline py-16 sm:py-24">
        <div className="mx-auto max-w-[1280px] px-5 sm:px-8">
          <SectionKicker index="06" title="docs tracks" meta="start anywhere" />
          <div className="mt-10 grid gap-8 lg:grid-cols-3">
            {docs.map(([title, text, href], index) => (
              <Link
                key={title}
                href={href}
                className="group edge-card border hairline p-6 transition-colors hover:border-[var(--rule-strong)]"
              >
                <div className="flex items-baseline justify-between gap-4">
                  <span className="mono-label text-[var(--accent)]">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                  <span className="font-mono text-[var(--fg-faint)] transition-transform duration-200 group-hover:translate-x-1 group-hover:text-[var(--accent)]">
                    -&gt;
                  </span>
                </div>
                <h2 className="mt-8 display text-[1.55rem] leading-tight text-[var(--fg)]">
                  {title}
                </h2>
                <p className="mt-4 text-[0.96rem] leading-relaxed text-[var(--fg-dim)]">{text}</p>
              </Link>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA band ─────────────────────────────────────────── */}
      <section className="relative overflow-clip py-20 sm:py-28">
        <span className="grid-bg pointer-events-none absolute inset-0 opacity-40" aria-hidden="true" />
        <div className="relative mx-auto max-w-[1280px] px-5 text-center sm:px-8">
          <p className="mono-label text-[var(--accent)]">reward logic as code</p>
          <h2 className="display mx-auto mt-6 max-w-[20ch] text-[clamp(2.2rem,1.6rem+2.4vw,4rem)] font-medium leading-[1.0]">
            Make the reward function a{" "}
            <span className="text-gradient">visible artifact.</span>
          </h2>
          <div className="mt-9 flex flex-col items-center gap-5">
            <InstallCommand className="w-full max-w-[30rem]" />
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Link href="/docs" className="button-link">
                Read the docs
              </Link>
              <Link href="/worked-examples/letter-env/setup" className="button-link">
                Letter World
              </Link>
              <a
                href="https://github.com/TristanBester/pycrm"
                target="_blank"
                rel="noreferrer"
                className="button-link"
              >
                GitHub ↗
              </a>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}
