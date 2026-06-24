import type { ReactNode } from "react";
import { anchors } from "@/lib/docs";

const icons: Record<string, ReactNode> = {
  code: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <path d="m8 9-3 3 3 3" />
      <path d="m16 9 3 3-3 3" />
      <path d="m13.5 7-3 10" />
    </svg>
  ),
  paper: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4H11v15H5.5A1.5 1.5 0 0 0 4 20.5z" />
      <path d="M20 5.5A1.5 1.5 0 0 0 18.5 4H13v15h5.5a1.5 1.5 0 0 1 1.5 1.5z" />
    </svg>
  ),
  demo: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      <rect x="4" y="8" width="16" height="11" rx="2" />
      <path d="M12 8V4" />
      <circle cx="9" cy="13" r="1" />
      <circle cx="15" cy="13" r="1" />
      <path d="M8 8h8" />
    </svg>
  ),
};

export function DocsAnchors() {
  return (
    <div className="mb-7 space-y-1">
      {anchors.map((anchor) => (
        <a
          key={anchor.label}
          href={anchor.href}
          target="_blank"
          rel="noreferrer"
          className="group flex items-center gap-3 border-l border-[var(--rule)] px-3 py-2 text-[13px] text-[var(--fg-faint)] transition-colors hover:border-[var(--rule-strong)] hover:bg-[color-mix(in_oklch,var(--bg-elev)_52%,transparent)] hover:text-[var(--fg)]"
        >
          <span className="text-[var(--accent)] transition-colors">{icons[anchor.icon]}</span>
          <span className="flex-1">{anchor.label}</span>
          <span className="font-mono text-[10px] text-[var(--fg-faint)] opacity-0 transition-opacity group-hover:opacity-100">
            &#8599;
          </span>
        </a>
      ))}
    </div>
  );
}
