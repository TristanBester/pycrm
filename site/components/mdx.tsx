import Link from "next/link";
import type { CSSProperties, ReactNode } from "react";

type Tone = "info" | "note" | "warning" | "tip";

const TONES: Record<Tone, { color: string; label: string }> = {
  info: { color: "var(--accent-2)", label: "Info" },
  note: { color: "var(--accent)", label: "Note" },
  warning: { color: "var(--warn)", label: "Warning" },
  tip: { color: "var(--accent-3)", label: "Tip" },
};

function CalloutIcon({ tone }: { tone: Tone }) {
  const common = {
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  if (tone === "warning") {
    return (
      <svg {...common}>
        <path d="M10.3 3.7 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.7a2 2 0 0 0-3.4 0Z" />
        <path d="M12 9v4" />
        <path d="M12 17h.01" />
      </svg>
    );
  }
  if (tone === "tip") {
    return (
      <svg {...common}>
        <path d="M9 18h6" />
        <path d="M10 22h4" />
        <path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V18h6v-1.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2Z" />
      </svg>
    );
  }
  if (tone === "note") {
    return (
      <svg {...common}>
        <path d="M12 20h9" />
        <path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" />
      </svg>
    );
  }
  return (
    <svg {...common}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5" />
      <path d="M12 8h.01" />
    </svg>
  );
}

function Box({
  children,
  tone,
  label,
}: {
  children?: ReactNode;
  tone: Tone;
  label?: string;
}) {
  const t = TONES[tone];
  return (
    <aside className="callout" style={{ "--tone": t.color } as CSSProperties}>
      <div className="callout-head">
        <CalloutIcon tone={tone} />
        <span className="callout-label">{label ?? t.label}</span>
      </div>
      <div className="callout-body">{children}</div>
    </aside>
  );
}

export function Info({ children }: { children?: ReactNode }) {
  return <Box tone="info">{children}</Box>;
}

export function Note({ children }: { children?: ReactNode }) {
  return <Box tone="note">{children}</Box>;
}

export function Warning({ children }: { children?: ReactNode }) {
  return <Box tone="warning">{children}</Box>;
}

export function Tip({ children }: { children?: ReactNode }) {
  return <Box tone="tip">{children}</Box>;
}

export function Callout({
  children,
  type = "info",
  title,
}: {
  children?: ReactNode;
  type?: Tone;
  title?: string;
}) {
  const tone: Tone = type in TONES ? type : "info";
  return (
    <Box tone={tone} label={title}>
      {children}
    </Box>
  );
}

/** Inline pill / tag for docs prose. */
const TAG_TONES: Record<string, string> = {
  accent: "var(--accent)",
  iris: "var(--accent-iris)",
  cyan: "var(--accent-2)",
  green: "var(--accent-3)",
  warn: "var(--warn)",
  neutral: "var(--fg-faint)",
};

export function Tag({
  children,
  tone = "accent",
}: {
  children?: ReactNode;
  tone?: keyof typeof TAG_TONES;
}) {
  const color = TAG_TONES[tone] ?? TAG_TONES.accent;
  return (
    <span className="doc-tag" style={{ "--tone": color } as CSSProperties}>
      {children}
    </span>
  );
}

export const Badge = Tag;

/** Numbered walkthrough. <Steps><Step title="…">…</Step></Steps> */
export function Steps({ children }: { children?: ReactNode }) {
  return <div className="doc-steps">{children}</div>;
}

export function Step({
  title,
  children,
}: {
  title?: string;
  children?: ReactNode;
}) {
  return (
    <div className="doc-step">
      {title ? <div className="doc-step-title">{title}</div> : null}
      <div className="doc-step-body">{children}</div>
    </div>
  );
}

export function CardGroup({
  children,
  cols = 2,
}: {
  children?: ReactNode;
  cols?: number;
}) {
  const colClass = cols >= 3 ? "sm:grid-cols-3" : cols === 1 ? "" : "sm:grid-cols-2";
  return <div className={`my-8 grid gap-4 ${colClass}`}>{children}</div>;
}

export function Card({
  title,
  href,
  children,
}: {
  title?: string;
  href?: string;
  children?: ReactNode;
}) {
  const className =
    "group block h-full edge-card border hairline p-5 no-underline transition-colors hover:border-[var(--rule-strong)]";
  const body = (
    <>
      <div className="flex items-baseline justify-between gap-4">
        <span className="font-display text-[1.2rem] leading-tight text-[var(--fg)]">{title}</span>
        <span className="font-mono text-[var(--fg-faint)] transition-transform duration-200 group-hover:translate-x-1 group-hover:text-[var(--accent)]">
          -&gt;
        </span>
      </div>
      {children ? (
        <div className="mt-3 text-[0.92rem] leading-relaxed text-[var(--fg-dim)] [&>p]:m-0">
          {children}
        </div>
      ) : null}
    </>
  );

  if (!href) {
    return <div className={className}>{body}</div>;
  }

  return href.startsWith("/") ? (
    <Link href={href} className={className}>
      {body}
    </Link>
  ) : (
    <a href={href} target="_blank" rel="noreferrer" className={className}>
      {body}
    </a>
  );
}

export function Frame({ children, caption }: { children?: ReactNode; caption?: string }) {
  return (
    <figure className="edge-card gradient-edge my-8 overflow-hidden border hairline shadow-[0_24px_60px_-30px_color-mix(in_oklch,var(--accent-iris)_30%,transparent)]">
      <div className="flex items-center gap-1.5 border-b hairline px-3 py-2">
        <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent)]" />
        <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent-iris)]" />
        <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent-2)]" />
        <span className="ml-auto mono-label text-[var(--fg-faint)]">pycrm capture</span>
      </div>
      <div className="p-3">{children}</div>
      {caption ? (
        <figcaption className="border-t hairline px-3 py-2 text-[12px] text-[var(--fg-faint)]">
          {caption}
        </figcaption>
      ) : null}
    </figure>
  );
}
