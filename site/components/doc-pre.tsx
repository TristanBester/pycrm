"use client";

import { useRef, useState, type ComponentProps } from "react";

/**
 * Wraps docs code blocks (the <pre> emitted by rehype-pretty-code) with a
 * hover copy button. Copies `innerText`, which excludes the CSS-generated line
 * numbers and language label (those are `::before` pseudo-content), so the
 * clipboard gets clean source.
 */
export function DocPre(props: ComponentProps<"pre">) {
  const ref = useRef<HTMLPreElement>(null);
  const [copied, setCopied] = useState(false);

  return (
    <div className="doc-code">
      <button
        type="button"
        className="doc-code-copy copy-btn"
        data-copied={copied || undefined}
        aria-label={copied ? "Copied" : "Copy code"}
        onClick={async () => {
          const text = ref.current?.innerText ?? "";
          try {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          } catch {
            /* clipboard unavailable — no-op */
          }
        }}
      >
        {copied ? "✓ copied" : "copy"}
      </button>
      <pre ref={ref} {...props} />
    </div>
  );
}
