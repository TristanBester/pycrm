export type Receipt = {
  label: string;
  metric: string;
  text: string;
};

export function Receipts({
  caption,
  rightLabel,
  rows,
  footnote,
}: {
  caption: string;
  rightLabel?: string;
  rows: Receipt[];
  footnote?: string;
}) {
  return (
    <div className="edge-card gradient-edge border hairline">
      <div className="flex items-center justify-between gap-4 border-b hairline px-5 py-3">
        <span className="mono-label text-[var(--fg-faint)]">{caption}</span>
        {rightLabel ? (
          <span className="mono-label text-[var(--accent)]">{rightLabel}</span>
        ) : null}
      </div>
      <div className="divide-y divide-[var(--rule)]">
        {rows.map((r) => (
          <div
            key={r.label}
            className="flex items-baseline justify-between gap-6 px-5 py-5"
          >
            <div>
              <div className="mono-label text-[var(--fg-faint)]">{r.label}</div>
              <p className="mt-2 max-w-[42ch] text-[0.9rem] leading-relaxed text-[var(--fg-dim)]">
                {r.text}
              </p>
            </div>
            <div className="display whitespace-nowrap text-[1.75rem] leading-none text-gradient">
              {r.metric}
            </div>
          </div>
        ))}
      </div>
      {footnote ? (
        <div className="border-t hairline px-5 py-2.5 mono-label text-[var(--fg-faint)]">
          {footnote}
        </div>
      ) : null}
    </div>
  );
}
