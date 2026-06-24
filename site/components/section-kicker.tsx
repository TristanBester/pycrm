export function SectionKicker({
  index,
  title,
  meta,
}: {
  index: string;
  title: string;
  meta?: string;
}) {
  return (
    <div className="flex items-baseline gap-3">
      <span className="mono-label text-[var(--accent)]">§ {index}</span>
      <span className="mono-label text-[var(--fg-faint)]">{title}</span>
      <span className="h-px flex-1 bg-[var(--rule)]" aria-hidden="true" />
      {meta ? (
        <span className="hidden mono-label text-[var(--fg-faint)] sm:inline">
          {meta}
        </span>
      ) : null}
    </div>
  );
}
