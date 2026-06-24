"use client";

import { useEffect, useState } from "react";

function applyTheme(theme: "dark" | "light") {
  document.documentElement.classList.toggle("light", theme === "light");
  document.documentElement.classList.toggle("dark", theme === "dark");
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("pycrm-theme", theme);
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    const stored = localStorage.getItem("pycrm-theme");
    const system = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    const next = stored === "light" || stored === "dark" ? stored : system;
    applyTheme(next);
    const frame = window.requestAnimationFrame(() => setTheme(next));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  const next = theme === "dark" ? "light" : "dark";

  return (
    <button
      type="button"
      aria-label={`Switch to ${next} theme`}
      aria-pressed={theme === "light"}
      title={`Switch to ${next} theme`}
      className="theme-toggle"
      onClick={() => {
        setTheme(next);
        applyTheme(next);
      }}
    >
      <span aria-hidden="true" className={`theme-option ${theme === "dark" ? "is-active" : ""}`}>
        dk
      </span>
      <span aria-hidden="true" className={`theme-option ${theme === "light" ? "is-active" : ""}`}>
        lt
      </span>
    </button>
  );
}
