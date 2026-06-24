import Link from "next/link";

const columns: { title: string; links: { label: string; href: string }[] }[] = [
  {
    title: "Documentation",
    links: [
      { label: "Get Started", href: "/docs" },
      { label: "Installation", href: "/installation" },
      { label: "Quickstart", href: "/quickstart" },
      { label: "API Reference", href: "/api-reference" },
    ],
  },
  {
    title: "Letter World",
    links: [
      { label: "Setup", href: "/worked-examples/letter-env/setup" },
      { label: "Cross-Product", href: "/worked-examples/letter-env/cross-product" },
      { label: "Q-Learning", href: "/worked-examples/letter-env/q-learning" },
      {
        label: "Counterfactual",
        href: "/worked-examples/letter-env/counterfactual-q-learning",
      },
    ],
  },
  {
    title: "Concepts",
    links: [
      { label: "Labelling Functions", href: "/core-concepts/labelling-functions" },
      { label: "Automata", href: "/core-concepts/automata" },
      { label: "Cross-Products", href: "/core-concepts/cross-products" },
      { label: "Agents", href: "/core-concepts/agents" },
    ],
  },
  {
    title: "Resources",
    links: [
      { label: "Paper ↗", href: "https://arxiv.org/abs/2312.11364" },
      { label: "Live Demo ↗", href: "https://crm.tristanbester.xyz" },
      { label: "GitHub ↗", href: "https://github.com/TristanBester/pycrm" },
      { label: "PyPI ↗", href: "https://pypi.org/project/pyrewardmachines" },
    ],
  },
];

export function SiteFooter() {
  return (
    <footer className="relative mt-24 border-t hairline">
      <span className="rule-gradient absolute inset-x-0 top-0" aria-hidden="true" />
      <div className="mx-auto grid max-w-[1280px] gap-12 px-5 py-16 sm:px-8 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,2fr)]">
        <div>
          <Link href="/" className="flex items-center gap-2.5">
            <img src="/favicon.svg" alt="PyCRM" className="h-9 w-9 rounded-[10px]" />
            <span className="brand text-[15px] font-extrabold tracking-[0.01em] text-[var(--fg)]">
              PyCRM
            </span>
          </Link>
          <p className="mt-5 max-w-[34ch] text-[0.92rem] leading-relaxed text-[var(--fg-dim)]">
            Counting Reward Machines for reinforcement learning — reward logic
            you can read, version, and exploit.
          </p>
          <p className="mt-5 mono-label text-[var(--fg-faint)]">
            MIT · Python ≥ 3.10 · Gymnasium
          </p>
        </div>
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-4">
          {columns.map((col) => (
            <div key={col.title}>
              <div className="mono-label text-[var(--fg-faint)]">{col.title}</div>
              <ul className="mt-4 space-y-2.5">
                {col.links.map((link) => {
                  const external = link.href.startsWith("http");
                  return (
                    <li key={link.label}>
                      {external ? (
                        <a
                          href={link.href}
                          target="_blank"
                          rel="noreferrer"
                          className="link text-[0.86rem] text-[var(--fg-dim)] transition-colors hover:text-[var(--fg)]"
                        >
                          {link.label}
                        </a>
                      ) : (
                        <Link
                          href={link.href}
                          className="link text-[0.86rem] text-[var(--fg-dim)] transition-colors hover:text-[var(--fg)]"
                        >
                          {link.label}
                        </Link>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      </div>
      <div className="border-t hairline">
        <div className="mx-auto flex max-w-[1280px] flex-col gap-2 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <span className="mono-label text-[var(--fg-faint)]">
            © 2025–2026 PyCRM · built for reward machines
          </span>
          <span className="mono-label text-[var(--fg-faint)]">
            reward logic <span className="text-gradient">as code</span>
          </span>
        </div>
      </div>
    </footer>
  );
}
