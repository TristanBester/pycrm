"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { SearchDialog } from "@/components/search-dialog";
import { ThemeToggle } from "@/components/theme-toggle";
import { docs } from "@/lib/docs";

const nav = [
  { href: "/docs", label: "Docs" },
  { href: "/quickstart", label: "Quickstart" },
  { href: "/worked-examples/letter-env/ground-environment", label: "Letter World" },
  { href: "https://arxiv.org/abs/2312.11364", label: "Paper" },
  { href: "https://crm.tristanbester.xyz", label: "Demo" },
  { href: "https://github.com/TristanBester/pycrm", label: "GitHub" },
];

const docPaths = new Set(docs.map((doc) => `/${doc.slug}`));

function isDocsRoute(pathname: string) {
  const normalized = pathname.replace(/\/+$/, "") || "/";
  return (
    normalized === "/docs" ||
    normalized.startsWith("/docs/") ||
    docPaths.has(normalized)
  );
}

export function SiteHeader() {
  const pathname = usePathname();
  // Keep the header aligned with the content beneath it: docs pages are wider
  // (1440px) than the marketing pages (1280px).
  const widthClass = isDocsRoute(pathname) ? "max-w-[1440px]" : "max-w-[1280px]";

  return (
    <header className="site-header sticky top-0 z-40 border-b hairline bg-[color-mix(in_oklch,var(--bg)_76%,transparent)] backdrop-blur-md">
      <div className={`mx-auto flex h-16 ${widthClass} items-center gap-4 px-5 sm:px-8`}>
        <Link href="/" className="group flex items-center gap-2.5">
          <span className="brand-mark relative inline-flex h-9 w-9">
            <img
              src="/favicon.svg"
              alt="PyCRM"
              className="h-9 w-9 rounded-[10px] transition-transform duration-300 group-hover:scale-[1.04]"
            />
          </span>
          <span className="brand text-[15px] font-extrabold tracking-[0.01em]">
            PyCRM
          </span>
        </Link>
        <nav className="ml-4 hidden items-center gap-4 lg:flex">
          {nav.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className="link font-mono text-[12px] text-[var(--fg-faint)] transition-colors hover:text-[var(--fg)]"
            >
              {item.label}
            </Link>
          ))}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <SearchDialog />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
