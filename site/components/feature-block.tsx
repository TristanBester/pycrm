import type { ReactNode } from "react";
import { SpecTable } from "./spec-table";

export function FeatureBlock({
  index,
  kicker,
  title,
  children,
  specs,
}: {
  index: string;
  kicker: string;
  title: string;
  children?: ReactNode;
  specs?: [string, string][];
}) {
  return (
    <div className="grid gap-6 border-t hairline py-10 lg:grid-cols-[6.5rem_minmax(0,1fr)_minmax(0,17rem)] lg:gap-10">
      <div>
        <div className="mono-label text-[var(--accent)]">§ {index}</div>
        <div className="mt-1.5 mono-label text-[var(--fg-faint)]">{kicker}</div>
      </div>
      <div>
        <h3 className="display text-[1.55rem] font-medium leading-tight text-[var(--fg)]">
          {title}
        </h3>
        <div className="mt-3 max-w-[54ch] text-[0.96rem] leading-relaxed text-[var(--fg-dim)]">
          {children}
        </div>
      </div>
      {specs ? (
        <SpecTable
          rows={specs}
          className="self-start lg:border-l lg:border-[var(--rule)] lg:pl-7"
        />
      ) : (
        <div aria-hidden="true" />
      )}
    </div>
  );
}
