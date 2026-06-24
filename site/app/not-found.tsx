import Link from "next/link";

export default function NotFound() {
  return (
    <main id="main" className="mx-auto flex min-h-[calc(82vh-4rem)] max-w-[1280px] flex-col justify-center px-5 py-24 sm:px-8">
      <p className="mono-label text-[var(--accent)]">404</p>
      <h1 className="display mt-6 max-w-[18ch] text-[clamp(2.1rem,1.5rem+2vw,3.4rem)] leading-[1]">
        This transition is undefined.
      </h1>
      <p className="mt-6 max-w-[52ch] text-[var(--fg-dim)]">
        The requested page is not in the PyCRM docs automaton. Return to the
        documentation index and pick a valid state.
      </p>
      <Link className="button-link mt-8" href="/docs">
        Docs index
      </Link>
    </main>
  );
}
