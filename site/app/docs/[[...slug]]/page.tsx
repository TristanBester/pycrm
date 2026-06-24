import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { DocsPage } from "@/components/docs-page";
import { docs, getDocBySlug, getIntroDoc, loadDocModule } from "@/lib/docs";

type PageProps = {
  params: Promise<{ slug?: string[] }>;
};

export function generateStaticParams() {
  return [{ slug: [] }, ...docs.map((doc) => ({ slug: doc.slug.split("/") }))];
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const doc = slug?.length ? getDocBySlug(slug.join("/")) : getIntroDoc();
  if (!doc) {
    return {};
  }
  return {
    title: doc.title,
    description: doc.description,
  };
}

export default async function Page({ params }: PageProps) {
  const { slug } = await params;
  const doc = slug?.length ? getDocBySlug(slug.join("/")) : getIntroDoc();
  if (!doc) {
    notFound();
  }
  const mod = await loadDocModule(doc.slug);
  const Content = mod.default;
  return (
    <DocsPage doc={doc}>
      <Content />
    </DocsPage>
  );
}
