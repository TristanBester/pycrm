import Link from "next/link";
import type { ReactNode } from "react";
import { DocsSidebar } from "@/components/docs-sidebar";
import { DocsToc } from "@/components/docs-toc";
import { getNextPrev, type Doc } from "@/lib/docs";

export function DocsPage({ doc, children }: { doc: Doc; children: ReactNode }) {
  const { prev, next } = getNextPrev(doc);
  return (
    <main id="main" className="mx-auto max-w-[1440px] px-5 py-12 sm:px-8">
      <div className="docs-grid-calm" aria-hidden="true" />
      <div className="grid gap-12 lg:grid-cols-[19rem_minmax(0,1fr)_15rem]">
        <DocsSidebar current={doc} />
        <article data-pagefind-body className="min-w-0">
          <header className="mb-10 border-b hairline pb-8">
            <div className="flex items-baseline gap-3">
              <span className="mono-label text-[var(--accent)]">{doc.index}</span>
              <span className="h-px flex-1 bg-[var(--rule)]" />
              <span className="mono-label text-[var(--fg-faint)]">{doc.group}</span>
            </div>
            <h1 className="display mt-7 max-w-[20ch] text-[clamp(2rem,1.4rem+1.6vw,3rem)] font-medium leading-[1.02]">
              {doc.title}
            </h1>
            <p className="mt-6 max-w-[58ch] text-[1.02rem] leading-relaxed text-[var(--fg-dim)]">
              {doc.description}
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              {doc.tags.map((tag) => (
                <span
                  key={tag}
                  className="accent-chip px-2 py-1 font-mono text-[11px] text-[var(--fg-faint)]"
                >
                  {tag}
                </span>
              ))}
            </div>
          </header>
          <div className="prose">{children}</div>
          <nav className="mt-14 grid gap-px border hairline bg-[var(--rule)] sm:grid-cols-2">
            {prev ? (
              <Link href={`/${prev.slug}`} className="edge-card bg-[var(--bg)] p-5 transition-colors hover:text-[var(--accent)]">
                <div className="mono-label text-[var(--fg-faint)]">previous</div>
                <div className="mt-2 font-display text-[1.25rem] text-[var(--fg)]">{prev.title}</div>
              </Link>
            ) : (
              <div className="bg-[var(--bg)] p-5" />
            )}
            {next ? (
              <Link href={`/${next.slug}`} className="edge-card bg-[var(--bg)] p-5 text-right transition-colors hover:text-[var(--accent)]">
                <div className="mono-label text-[var(--fg-faint)]">next</div>
                <div className="mt-2 font-display text-[1.25rem] text-[var(--fg)]">{next.title}</div>
              </Link>
            ) : (
              <div className="bg-[var(--bg)] p-5" />
            )}
          </nav>
        </article>
        <aside className="hidden lg:block">
          <div className="sticky top-24">
            <div className="mono-label mb-3 pl-3 text-[var(--fg-faint)]">on this page</div>
            <DocsToc items={doc.toc} />
          </div>
        </aside>
      </div>
    </main>
  );
}
