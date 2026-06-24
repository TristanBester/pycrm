"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

type PagefindResult = {
  id: string;
  data: () => Promise<{
    url: string;
    meta: { title?: string };
    excerpt: string;
  }>;
};

type PagefindModule = {
  search: (query: string) => Promise<{ results: PagefindResult[] }>;
  options?: (options: { excerptLength: number }) => Promise<void>;
};

declare global {
  interface Window {
    pagefind?: PagefindModule;
  }
}

export function SearchDialog() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("Type to search the rendered docs.");
  const [results, setResults] = useState<
    Array<{ id: string; title: string; url: string; excerpt: string }>
  >([]);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
      }
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 0);
    }
  }, [open]);

  useEffect(() => {
    let cancelled = false;
    async function runSearch() {
      const trimmed = query.trim();
      if (!trimmed) {
        setResults([]);
        setStatus("Type to search the rendered docs.");
        return;
      }
      setStatus("Searching...");
      try {
        const pagefind =
          window.pagefind ??
          ((await new Function(
            'return import("/pagefind/pagefind.js")',
          )()) as PagefindModule);
        window.pagefind = pagefind;
        await pagefind.options?.({ excerptLength: 18 });
        const response = await pagefind.search(trimmed);
        const hydrated = await Promise.all(
          response.results.slice(0, 8).map(async (result) => {
            const data = await result.data();
            return {
              id: result.id,
              title: data.meta.title ?? "Untitled",
              url: data.url,
              excerpt: data.excerpt,
            };
          }),
        );
        if (!cancelled) {
          setResults(hydrated);
          setStatus(hydrated.length ? `${hydrated.length} result${hydrated.length === 1 ? "" : "s"}.` : "No results.");
        }
      } catch {
        if (!cancelled) {
          setResults([]);
          setStatus("Search index is available after `npm run build`.");
        }
      }
    }
    const timeout = window.setTimeout(runSearch, 120);
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
    };
  }, [query]);

  return (
    <>
      <button
        type="button"
        className="hidden min-w-[14rem] items-center justify-between border hairline bg-[color-mix(in_oklch,var(--bg)_72%,transparent)] px-3 py-2 text-left text-[12px] text-[var(--fg-faint)] backdrop-blur-sm transition-colors hover:text-[var(--fg)] md:inline-flex"
        onClick={() => setOpen(true)}
      >
        <span className="font-mono">Search docs</span>
        <span className="font-mono text-[10px] uppercase tracking-[0.16em]">Ctrl K</span>
      </button>
      <button
        type="button"
        className="inline-flex h-10 w-10 items-center justify-center border hairline bg-[var(--bg-elev)] font-mono text-[13px] text-[var(--fg-dim)] transition-colors hover:text-[var(--accent)] md:hidden"
        aria-label="Search docs"
        onClick={() => setOpen(true)}
      >
        /
      </button>
      {open ? (
        <div
          className="fixed inset-0 z-50 bg-black/45 px-4 py-16 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-label="Search PyCRM docs"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              setOpen(false);
            }
          }}
        >
          <div className="mx-auto max-w-2xl border hairline bg-[var(--bg)] shadow-[0_30px_100px_-40px_var(--accent)]">
            <div className="border-b hairline p-4">
              <input
                ref={inputRef}
                className="w-full bg-transparent font-mono text-[1rem] text-[var(--fg)] outline-none placeholder:text-[var(--fg-faint)]"
                placeholder="Search reward machines, labels, agents..."
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
            <div className="max-h-[24rem] overflow-y-auto p-2">
              <p className="px-3 py-2 font-mono text-[11px] uppercase tracking-[0.18em] text-[var(--fg-faint)]">
                {status}
              </p>
              <div className="divide-y divide-[var(--rule)]">
                {results.map((result) => (
                  <Link
                    key={result.id}
                    href={result.url}
                    className="block px-3 py-3 transition-colors hover:bg-[var(--bg-elev)]"
                    onClick={() => setOpen(false)}
                  >
                    <div className="font-display text-[1.15rem] text-[var(--fg)]">
                      {result.title}
                    </div>
                    <p
                      className="mt-1 text-[13px] leading-relaxed text-[var(--fg-dim)]"
                      dangerouslySetInnerHTML={{ __html: result.excerpt }}
                    />
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
