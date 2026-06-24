import type { ComponentType, ReactNode } from "react";
import {
  Badge,
  Callout,
  Card,
  CardGroup,
  Frame,
  Info,
  Note,
  Step,
  Steps,
  Tag,
  Tip,
  Warning,
} from "@/components/mdx";
import { DocPre } from "@/components/doc-pre";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type MDXComponents = Record<string, ComponentType<any>>;

function slugify(value: unknown) {
  return String(value)
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

function Heading({
  level,
  children,
}: {
  level: 2 | 3;
  children?: ReactNode;
}) {
  const id = slugify(children);
  const Tag = `h${level}` as "h2" | "h3";
  return (
    <Tag id={id}>
      <a href={`#${id}`} className="heading-anchor" aria-label={`Link to ${String(children)}`}>
        {children}
      </a>
    </Tag>
  );
}

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    h2: ({ children }) => <Heading level={2}>{children}</Heading>,
    h3: ({ children }) => <Heading level={3}>{children}</Heading>,
    pre: DocPre,
    Info,
    Note,
    Warning,
    Tip,
    Callout,
    Tag,
    Badge,
    Steps,
    Step,
    Frame,
    Card,
    CardGroup,
    ...components,
  };
}
