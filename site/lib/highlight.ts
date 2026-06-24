import { codeToHtml } from "shiki";

/**
 * Server-side syntax highlighting for the marketing pages, using the same
 * Shiki themes as the docs (rehype-pretty-code). `defaultColor: false` emits
 * `--shiki-dark` / `--shiki-light` CSS variables per token so the existing
 * theme-switch CSS in globals.css picks the right color. Runs at build time
 * (static export) — never ships Shiki to the client.
 */
export async function highlight(code: string, lang: string): Promise<string> {
  return codeToHtml(code.trimEnd(), {
    lang,
    themes: { dark: "github-dark-dimmed", light: "github-light" },
    defaultColor: false,
  });
}
