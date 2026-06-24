export function SpecTable({
  rows,
  className = "",
}: {
  rows: [string, string][];
  className?: string;
}) {
  return (
    <dl className={`spec-table ${className}`}>
      {rows.map(([key, value]) => (
        <div
          key={key}
          className="grid grid-cols-[auto_minmax(0,1fr)] items-baseline gap-4 border-t hairline py-2.5 first:border-t-0"
        >
          <dt className="mono-label text-[var(--fg-faint)]">{key}</dt>
          <dd className="text-right font-mono text-[0.78rem] leading-snug text-[var(--fg-dim)]">
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
