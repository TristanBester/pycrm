import type { ComponentType } from "react";

export type TocItem = {
  id: string;
  label: string;
};

export type TabName = "Documentation" | "API Reference";

export type DocGroup =
  | "Get Started"
  | "Letter World Tutorial"
  | "Core Concepts"
  | "API Reference";

export type Doc = {
  slug: string;
  title: string;
  description: string;
  tab: TabName;
  group: DocGroup;
  index: string;
  tags: string[];
  toc: TocItem[];
};

type MdxModule = {
  default: ComponentType;
};

export const docs = [
  {
    slug: "introduction",
    title: "Introduction",
    description: "Reward Machines, Counting Reward Machines, and the shape of PyCRM.",
    tab: "Documentation",
    group: "Get Started",
    index: "01",
    tags: ["overview", "rm", "crm"],
    toc: [
      { id: "what-pycrm-solves", label: "What PyCRM Solves" },
      { id: "reward-machines", label: "Reward Machines" },
      { id: "counting-reward-machines", label: "Counting Reward Machines" },
      { id: "research-map", label: "Research Map" },
    ],
  },
  {
    slug: "installation",
    title: "Installation",
    description: "Install PyCRM from PyPI or from source for research work.",
    tab: "Documentation",
    group: "Get Started",
    index: "02",
    tags: ["install", "python", "uv"],
    toc: [
      { id: "requirements", label: "Requirements" },
      { id: "install-from-pypi", label: "Install From PyPI" },
      { id: "development-install", label: "Development Install" },
      { id: "verify-the-install", label: "Verify The Install" },
    ],
  },
  {
    slug: "quickstart",
    title: "Quick Start",
    description: "Build a Letter World task from environment, labels, machine, and cross-product.",
    tab: "Documentation",
    group: "Get Started",
    index: "03",
    tags: ["quickstart", "letter world"],
    toc: [
      { id: "the-four-pieces", label: "The Four Pieces" },
      { id: "minimal-example", label: "Minimal Example" },
      { id: "what-the-wrapper-adds", label: "What The Wrapper Adds" },
      { id: "train-a-tabular-agent", label: "Train A Tabular Agent" },
    ],
  },
  {
    slug: "worked-examples/letter-env/setup",
    title: "Setup & Overview",
    description: "Prepare the repository examples and understand the tutorial path.",
    tab: "Documentation",
    group: "Letter World Tutorial",
    index: "04",
    tags: ["examples", "setup", "uv"],
    toc: [
      { id: "clone-the-repository", label: "Clone The Repository" },
      { id: "install-the-example-stack", label: "Install The Example Stack" },
      { id: "example-map", label: "Example Map" },
      { id: "run-the-introduction-scripts", label: "Run The Scripts" },
    ],
  },
  {
    slug: "worked-examples/letter-env/ground-environment",
    title: "1 - Ground Environment",
    description: "The Gymnasium grid world before reward-machine structure is added.",
    tab: "Documentation",
    group: "Letter World Tutorial",
    index: "05",
    tags: ["letter world", "gymnasium", "environment"],
    toc: [
      { id: "world-layout", label: "World Layout" },
      { id: "observations-and-actions", label: "Observations And Actions" },
      { id: "movement", label: "Movement" },
      { id: "stochastic-letter-state", label: "Stochastic Letter State" },
    ],
  },
  {
    slug: "worked-examples/letter-env/labelling-function",
    title: "2 - Labelling Function",
    description: "Extract symbolic events from Letter World transitions.",
    tab: "Documentation",
    group: "Letter World Tutorial",
    index: "06",
    tags: ["letter world", "labels"],
    toc: [
      { id: "symbol-vocabulary", label: "Symbol Vocabulary" },
      { id: "event-methods", label: "Event Methods" },
      { id: "using-the-labeler", label: "Using The Labeler" },
      { id: "failure-modes", label: "Failure Modes" },
    ],
  },
  {
    slug: "worked-examples/letter-env/crm",
    title: "3 - Reward Machine",
    description: "Encode the Letter World objective as an RM, then extend it with counters.",
    tab: "Documentation",
    group: "Letter World Tutorial",
    index: "07",
    tags: ["letter world", "crm", "rm"],
    toc: [
      { id: "sequential-rm", label: "Sequential RM" },
      { id: "transition-tables", label: "Transition Tables" },
      { id: "why-counting-matters", label: "Why Counting Matters" },
      { id: "balanced-task-sketch", label: "Balanced Task Sketch" },
    ],
  },
  {
    slug: "worked-examples/letter-env/cross-product",
    title: "4 - Cross-Product",
    description: "Wrap Letter World with the machine and expose a standard Gymnasium API.",
    tab: "Documentation",
    group: "Letter World Tutorial",
    index: "08",
    tags: ["letter world", "cross-product"],
    toc: [
      { id: "assemble-the-wrapper", label: "Assemble The Wrapper" },
      { id: "step-like-gymnasium", label: "Step Like Gymnasium" },
      { id: "read-the-observation", label: "Read The Observation" },
      { id: "run-an-episode", label: "Run An Episode" },
    ],
  },
  {
    slug: "worked-examples/letter-env/q-learning",
    title: "5 - Q-Learning in Letter World",
    description: "Train a tabular baseline on the cross-product environment.",
    tab: "Documentation",
    group: "Letter World Tutorial",
    index: "09",
    tags: ["letter world", "q-learning"],
    toc: [
      { id: "baseline-loop", label: "Baseline Loop" },
      { id: "training-curve", label: "Training Curve" },
      { id: "inspect-the-policy", label: "Inspect The Policy" },
      { id: "limits-of-the-baseline", label: "Limits Of The Baseline" },
    ],
  },
  {
    slug: "worked-examples/letter-env/counterfactual-q-learning",
    title: "6 - Counterfactual Q-Learning",
    description: "Use CRM structure to update many machine states from one real transition.",
    tab: "Documentation",
    group: "Letter World Tutorial",
    index: "10",
    tags: ["letter world", "counterfactuals", "cql"],
    toc: [
      { id: "the-counterfactual-idea", label: "The Counterfactual Idea" },
      { id: "standard-vs-counterfactual", label: "Standard Vs Counterfactual" },
      { id: "learning-curve", label: "Learning Curve" },
      { id: "when-it-helps", label: "When It Helps" },
    ],
  },
  {
    slug: "core-concepts/labelling-functions",
    title: "Labelling Functions",
    description: "Turn low-level transitions into symbolic events.",
    tab: "Documentation",
    group: "Core Concepts",
    index: "11",
    tags: ["labels", "events", "symbols"],
    toc: [
      { id: "why-labels-exist", label: "Why Labels Exist" },
      { id: "event-detectors", label: "Event Detectors" },
      { id: "letter-world-labels", label: "Letter World Labels" },
      { id: "design-rules", label: "Design Rules" },
    ],
  },
  {
    slug: "core-concepts/automata",
    title: "RMs & CRMs",
    description: "Define task progress and rewards with automata.",
    tab: "Documentation",
    group: "Core Concepts",
    index: "12",
    tags: ["automata", "reward machines", "counters"],
    toc: [
      { id: "automata-as-reward-logic", label: "Automata As Reward Logic" },
      { id: "rm-vs-crm", label: "RM Vs CRM" },
      { id: "transition-functions", label: "Transition Functions" },
      { id: "counter-conditions", label: "Counter Conditions" },
    ],
  },
  {
    slug: "core-concepts/cross-products",
    title: "Cross-Product Environments",
    description: "Combine environment dynamics with reward-machine state.",
    tab: "Documentation",
    group: "Core Concepts",
    index: "13",
    tags: ["cross-product", "gymnasium", "observations"],
    toc: [
      { id: "the-product-state", label: "The Product State" },
      { id: "step-flow", label: "Step Flow" },
      { id: "observation-shapes", label: "Observation Shapes" },
      { id: "counterfactual-hook", label: "Counterfactual Hook" },
    ],
  },
  {
    slug: "core-concepts/agents",
    title: "Reinforcement Learning Agents",
    description: "Tabular and deep RL agents that consume counterfactual experience.",
    tab: "Documentation",
    group: "Core Concepts",
    index: "14",
    tags: ["agents", "q-learning", "sb3", "sbx"],
    toc: [
      { id: "agent-families", label: "Agent Families" },
      { id: "tabular-agents", label: "Tabular Agents" },
      { id: "deep-rl-agents", label: "Deep RL Agents" },
      { id: "when-to-use-counterfactuals", label: "When To Use Counterfactuals" },
    ],
  },
  {
    slug: "api-reference",
    title: "API Reference",
    description: "Generated reference for the PyCRM package modules and classes.",
    tab: "API Reference",
    group: "API Reference",
    index: "01",
    tags: ["reference", "api"],
    toc: [
      { id: "status", label: "Status" },
      { id: "what-it-will-cover", label: "What It Will Cover" },
      { id: "in-the-meantime", label: "In The Meantime" },
    ],
  },
] satisfies Doc[];

export const tabs: { name: TabName; groups: DocGroup[] }[] = [
  {
    name: "Documentation",
    groups: ["Get Started", "Letter World Tutorial", "Core Concepts"],
  },
  {
    name: "API Reference",
    groups: ["API Reference"],
  },
];

export const tabNames = tabs.map((tab) => tab.name);

export const docGroups = tabs[0].groups;

export const anchors = [
  {
    label: "Code",
    href: "https://github.com/TristanBester/pycrm",
    icon: "code" as const,
  },
  {
    label: "Paper",
    href: "https://arxiv.org/abs/2312.11364",
    icon: "paper" as const,
  },
  {
    label: "Demo",
    href: "https://crm.tristanbester.xyz",
    icon: "demo" as const,
  },
];

const docImporters: Record<string, () => Promise<MdxModule>> = {
  introduction: () => import("@/content/docs/introduction.mdx"),
  installation: () => import("@/content/docs/installation.mdx"),
  quickstart: () => import("@/content/docs/quickstart.mdx"),
  "core-concepts/labelling-functions": () =>
    import("@/content/docs/core-concepts/labelling-functions.mdx"),
  "core-concepts/automata": () => import("@/content/docs/core-concepts/automata.mdx"),
  "core-concepts/cross-products": () =>
    import("@/content/docs/core-concepts/cross-products.mdx"),
  "core-concepts/agents": () => import("@/content/docs/core-concepts/agents.mdx"),
  "worked-examples/letter-env/setup": () =>
    import("@/content/docs/worked-examples/letter-env/setup.mdx"),
  "worked-examples/letter-env/ground-environment": () =>
    import("@/content/docs/worked-examples/letter-env/ground-environment.mdx"),
  "worked-examples/letter-env/labelling-function": () =>
    import("@/content/docs/worked-examples/letter-env/labelling-function.mdx"),
  "worked-examples/letter-env/crm": () =>
    import("@/content/docs/worked-examples/letter-env/crm.mdx"),
  "worked-examples/letter-env/cross-product": () =>
    import("@/content/docs/worked-examples/letter-env/cross-product.mdx"),
  "worked-examples/letter-env/q-learning": () =>
    import("@/content/docs/worked-examples/letter-env/q-learning.mdx"),
  "worked-examples/letter-env/counterfactual-q-learning": () =>
    import("@/content/docs/worked-examples/letter-env/counterfactual-q-learning.mdx"),
  "api-reference": () => import("@/content/docs/api-reference.mdx"),
};

export function getDocBySlug(slug: string) {
  return docs.find((doc) => doc.slug === slug);
}

export function getIntroDoc() {
  return docs[0];
}

export function getTabEntryDoc(tab: TabName) {
  return docs.find((doc) => doc.tab === tab);
}

export function getNextPrev(doc: Doc) {
  const sameTab = docs.filter((item) => item.tab === doc.tab);
  const index = sameTab.findIndex((item) => item.slug === doc.slug);
  return {
    prev: index > 0 ? sameTab[index - 1] : undefined,
    next: index >= 0 && index < sameTab.length - 1 ? sameTab[index + 1] : undefined,
  };
}

export function getDocsByGroup(group: Doc["group"]) {
  return docs.filter((doc) => doc.group === group);
}

export async function loadDocModule(slug: string) {
  const importer = docImporters[slug];
  if (!importer) {
    throw new Error(`Unknown doc slug: ${slug}`);
  }
  return importer();
}
