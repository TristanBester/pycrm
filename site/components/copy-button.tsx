"use client";

import { useState } from "react";

export function CopyButton({
  value,
  className = "",
}: {
  value: string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);

  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : "Copy to clipboard"}
      data-copied={copied || undefined}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1400);
        } catch {
          /* clipboard unavailable — no-op */
        }
      }}
      className={`copy-btn ${className}`}
    >
      <span aria-hidden="true">{copied ? "✓ copied" : "copy"}</span>
    </button>
  );
}
