"use client";

import { useEffect, useState } from "react";
import type { TocItem } from "@/lib/docs";

export function DocsToc({ items }: { items: TocItem[] }) {
  const [active, setActive] = useState<string>(items[0]?.id ?? "");

  useEffect(() => {
    if (!items.length) return;

    const headings = items
      .map((item) => document.getElementById(item.id))
      .filter((el): el is HTMLElement => Boolean(el));

    if (!headings.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) {
          setActive(visible[0].target.id);
        }
      },
      { rootMargin: "-96px 0px -66% 0px", threshold: [0, 1] },
    );

    headings.forEach((heading) => observer.observe(heading));
    return () => observer.disconnect();
  }, [items]);

  if (!items.length) return null;

  return (
    <nav aria-label="On this page" className="space-y-1.5">
      {items.map((item) => {
        const isActive = item.id === active;
        return (
          <a
            key={item.id}
            href={`#${item.id}`}
            aria-current={isActive ? "location" : undefined}
            className={`block border-l py-1 pl-3 text-[12px] leading-snug transition-colors ${
              isActive
                ? "border-[var(--accent)] text-[var(--fg)]"
                : "border-transparent text-[var(--fg-faint)] hover:text-[var(--fg)]"
            }`}
          >
            {item.label}
          </a>
        );
      })}
    </nav>
  );
}
