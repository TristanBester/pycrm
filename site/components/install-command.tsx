"use client";

import { useState } from "react";
import { CopyButton } from "./copy-button";

const MANAGERS = [
  { id: "pip", label: "pip", cmd: "pip install pyrewardmachines" },
  { id: "uv", label: "uv", cmd: "uv add pyrewardmachines" },
  { id: "poetry", label: "poetry", cmd: "poetry add pyrewardmachines" },
  { id: "conda", label: "conda", cmd: "conda install -c conda-forge pyrewardmachines" },
] as const;

export function InstallCommand({ className = "" }: { className?: string }) {
  const [active, setActive] = useState<string>(MANAGERS[0].id);
  const current = MANAGERS.find((m) => m.id === active) ?? MANAGERS[0];
  const [bin, ...rest] = current.cmd.split(" ");

  return (
    <div className={`install-widget edge-card gradient-edge border hairline ${className}`}>
      <div className="flex items-center gap-1 border-b hairline px-2 py-1.5">
        {MANAGERS.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setActive(m.id)}
            className="seg-tab"
            data-active={m.id === active || undefined}
          >
            {m.label}
          </button>
        ))}
        <span className="ml-auto pr-1.5 mono-label text-[var(--fg-faint)]">PyPI</span>
      </div>
      <div className="flex items-center gap-3 px-4 py-3.5 font-mono text-[0.84rem] leading-none">
        <span className="text-[var(--accent-2)]">$</span>
        <code className="flex-1 overflow-x-auto whitespace-nowrap text-[var(--fg)]">
          <span className="text-[var(--accent)]">{bin}</span> {rest.join(" ")}
        </code>
        <CopyButton value={current.cmd} />
      </div>
    </div>
  );
}
