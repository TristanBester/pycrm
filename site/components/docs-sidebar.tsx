import Link from "next/link";
import { DocsAnchors } from "@/components/docs-anchors";
import { docs, getTabEntryDoc, tabs, type Doc } from "@/lib/docs";

export function DocsSidebar({ current }: { current?: Doc }) {
  const activeTab = current?.tab ?? "Documentation";
  const activeGroups = tabs.find((tab) => tab.name === activeTab)?.groups ?? [];

  return (
    <aside className="docs-sidebar hidden max-h-[calc(100vh-5rem)] overflow-y-auto pr-6 lg:block">
      <div className="sticky top-24">
        <nav className="mb-7 flex flex-wrap gap-1.5" aria-label="Documentation sections">
          {tabs.map((tab) => {
            const entry = getTabEntryDoc(tab.name);
            const isActive = tab.name === activeTab;
            return (
              <Link
                key={tab.name}
                href={entry ? `/${entry.slug}` : "/docs"}
                className={`mono-label border px-3 py-1.5 transition-colors ${
                  isActive
                    ? "border-[var(--accent)] bg-[color-mix(in_oklch,var(--accent)_12%,var(--bg-elev))] text-[var(--accent)]"
                    : "border-[var(--rule)] text-[var(--fg-faint)] hover:border-[var(--rule-strong)] hover:text-[var(--fg)]"
                }`}
              >
                {tab.name}
              </Link>
            );
          })}
        </nav>

        <DocsAnchors />

        {activeGroups.map((group) => (
          <div key={group} className="mb-8">
            <div className="mono-label mb-3 text-[var(--fg-faint)]">{group}</div>
            <div className="space-y-1.5">
              {docs
                .filter((doc) => doc.group === group)
                .map((doc) => (
                  <Link
                    key={doc.slug}
                    href={`/${doc.slug}`}
                    className={`grid grid-cols-[2.1rem_minmax(0,1fr)] gap-3 border-l px-3 py-2 text-[13px] transition-colors ${
                      current?.slug === doc.slug
                        ? "border-[var(--accent)] bg-[color-mix(in_oklch,var(--accent)_9%,var(--bg-elev))] text-[var(--fg)] shadow-[inset_1px_0_0_var(--accent)]"
                        : "border-[var(--rule)] text-[var(--fg-faint)] hover:border-[var(--rule-strong)] hover:bg-[color-mix(in_oklch,var(--bg-elev)_52%,transparent)] hover:text-[var(--fg)]"
                    }`}
                  >
                    <span className="font-mono text-[10px] text-[var(--accent)]">{doc.index}</span>
                    <span>{doc.title}</span>
                  </Link>
                ))}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
