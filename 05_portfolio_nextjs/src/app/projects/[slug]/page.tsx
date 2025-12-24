import Image from "next/image";
import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { Badge } from "@/components/badge";
import { Container } from "@/components/container";
import { getProjectBySlug, projects } from "@/data/projects";

export function generateStaticParams() {
  return projects.map((p) => ({ slug: p.slug }));
}

export function generateMetadata({
  params,
}: {
  params: { slug: string };
}): Metadata {
  const project = getProjectBySlug(params.slug);

  if (!project) {
    return {
      title: "Projet introuvable",
    };
  }

  return {
    title: `${project.title} — Soboure BELLO`,
    description: project.subtitle,
  };
}

export default function ProjectDetailPage({
  params,
}: {
  params: { slug: string };
}) {
  const project = getProjectBySlug(params.slug);

  if (!project) {
    notFound();
  }

  return (
    <div className="min-h-screen">
      <section className="py-14 sm:py-20">
        <Container>
          <div className="flex flex-col gap-10">
            <div className="space-y-4">
              <Link
                href="/projects"
                className="inline-flex items-center text-sm text-zinc-300 hover:text-white"
              >
                ← Back to projects
              </Link>
              <p className="text-xs font-medium text-zinc-400">{project.year}</p>
              <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                {project.title}
              </h1>
              <p className="max-w-3xl text-base leading-7 text-zinc-300">
                {project.subtitle}
              </p>

              <div className="flex flex-wrap gap-2 pt-2">
                {project.stack.map((t) => (
                  <Badge key={t}>{t}</Badge>
                ))}
              </div>
            </div>

            {project.coverImage ? (
              <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5">
                <div className="relative aspect-[16/8] w-full">
                  <Image
                    src={project.coverImage.src}
                    alt={project.coverImage.alt}
                    fill
                    className="object-cover"
                    sizes="(max-width: 1024px) 100vw, 1024px"
                    priority
                  />
                </div>
              </div>
            ) : null}

            <div className="grid gap-6 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                  <h2 className="text-lg font-semibold text-white">Highlights</h2>
                  <ul className="mt-4 list-disc space-y-2 pl-5 text-sm leading-6 text-zinc-300">
                    {project.highlights.map((h) => (
                      <li key={h}>{h}</li>
                    ))}
                  </ul>
                </div>
              </div>

              <div className="space-y-6">
                <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                  <h2 className="text-lg font-semibold text-white">Metrics</h2>
                  <div className="mt-4 space-y-3">
                    {project.metrics.map((m) => (
                      <div
                        key={m.label}
                        className="flex items-center justify-between gap-4 rounded-xl border border-white/10 bg-black/30 px-4 py-3"
                      >
                        <p className="text-xs text-zinc-400">{m.label}</p>
                        <p className="text-sm font-medium text-zinc-100">{m.value}</p>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                  <h2 className="text-lg font-semibold text-white">Links</h2>
                  <div className="mt-4 flex flex-col gap-3">
                    {project.links.map((l) => (
                      <a
                        key={l.href}
                        href={l.href}
                        className="inline-flex items-center justify-between rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-zinc-200 hover:bg-white/10 hover:text-white"
                        target={l.href.startsWith("http") ? "_blank" : undefined}
                        rel={l.href.startsWith("http") ? "noopener noreferrer" : undefined}
                      >
                        <span>{l.label}</span>
                        <span className="text-zinc-500">↗</span>
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Container>
      </section>
    </div>
  );
}
