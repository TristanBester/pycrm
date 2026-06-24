"use client";

import { useState } from "react";
import { CopyButton } from "./copy-button";

export type CodeTab = {
  label: string;
  lang: string;
  /** Pre-highlighted HTML from lib/highlight.ts (server-rendered). */
  html: string;
  /** Raw source, for the copy button. */
  raw: string;
};

export function CodeWindow({
  tabs,
  title = "letter_world.py",
}: {
  tabs: CodeTab[];
  title?: string;
}) {
  const [i, setI] = useState(0);
  const tab = tabs[i] ?? tabs[0];

  return (
    <div className="code-window edge-card gradient-edge border hairline">
      <div className="flex items-center gap-2 border-b hairline px-3 py-2">
        <span className="flex items-center gap-1.5">
          <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent)]" />
          <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent-iris)]" />
          <span className="h-[9px] w-[9px] rounded-full bg-[var(--accent-2)]" />
        </span>
        {tabs.length > 1 ? (
          <div className="ml-2 flex items-center gap-0.5 overflow-x-auto">
            {tabs.map((t, idx) => (
              <button
                key={t.label}
                type="button"
                onClick={() => setI(idx)}
                className="seg-tab"
                data-active={idx === i || undefined}
              >
                {t.label}
              </button>
            ))}
          </div>
        ) : (
          <span className="ml-2 mono-label text-[var(--fg-faint)]">{title}</span>
        )}
        <span className="ml-auto flex items-center gap-3">
          <span className="mono-label text-[var(--fg-faint)]">{tab.lang}</span>
          <CopyButton value={tab.raw} />
        </span>
      </div>
      <div
        className="code-window-body"
        dangerouslySetInnerHTML={{ __html: tab.html }}
      />
    </div>
  );
}
