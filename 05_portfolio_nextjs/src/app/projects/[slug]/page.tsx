import Image from "next/image";
import Link from "next/link";
import type { Metadata } from "next";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type React from "react";
import { readFile } from "node:fs/promises";
import path from "node:path";

import { Badge } from "@/components/badge";
import { Container } from "@/components/container";
import { getProjectBySlug, projects } from "@/data/projects";

function normalizeSlug(slug: string) {
  try {
    return decodeURIComponent(slug).trim().toLowerCase();
  } catch {
    return slug.trim().toLowerCase();
  }
}

async function fetchReadmeMarkdown(readmeUrl: string) {
  const res = await fetch(readmeUrl, {
    cache: "force-cache",
  });

  if (!res.ok) {
    return null;
  }

  return res.text();
}

async function readLocalReadmeMarkdown(readmePath: string) {
  const absolutePath = path.resolve(process.cwd(), readmePath);
  return readFile(absolutePath, "utf8");
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string }>;
}): Promise<Metadata> {
  const { slug } = await params;
  const project = getProjectBySlug(normalizeSlug(slug));

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

export default async function ProjectDetailPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const project = getProjectBySlug(normalizeSlug(slug));

  if (!project) {
    return (
      <div className="min-h-screen">
        <section className="py-14 sm:py-20">
          <Container>
            <div className="flex flex-col gap-6">
              <Link
                href="/projects"
                className="inline-flex items-center text-sm text-zinc-300 hover:text-white"
              >
                ← Back to projects
              </Link>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">
                  Projet introuvable
                </h1>
                <p className="mt-2 text-sm leading-6 text-zinc-300">
                  Ce projet n’existe pas pour le slug: <code>{slug}</code>
                </p>
                <p className="mt-4 text-sm text-zinc-300">Projets disponibles :</p>
                <div className="mt-3 grid gap-3 sm:grid-cols-2">
                  {projects.map((p) => (
                    <Link
                      key={p.slug}
                      href={`/projects/${p.slug}`}
                      className="rounded-xl border border-white/10 bg-black/30 px-4 py-3 text-sm text-zinc-200 hover:bg-white/10 hover:text-white"
                    >
                      {p.title}
                    </Link>
                  ))}
                </div>
              </div>
            </div>
          </Container>
        </section>
      </div>
    );
  }

  return <ProjectDetail project={project} />;
}

async function ProjectDetail({
  project,
}: {
  project: NonNullable<ReturnType<typeof getProjectBySlug>>;
}) {
  const assetBaseUrl = `/api/project-assets/${project.slug}`;
  let readmeMarkdown: string | null = null;

  if (project.readmePath) {
    try {
      readmeMarkdown = await readLocalReadmeMarkdown(project.readmePath);
    } catch {
      readmeMarkdown = null;
    }
  }

  if (!readmeMarkdown && project.readmeUrl) {
    readmeMarkdown = await fetchReadmeMarkdown(project.readmeUrl);
  }

  const githubLink = project.links.find((l) => l.label.includes("GitHub"));

  const markdownComponents: Components = {
    h1: (props: React.ComponentPropsWithoutRef<"h1">) => (
      <h3 {...props} className="text-lg font-semibold text-white" />
    ),
    h2: (props: React.ComponentPropsWithoutRef<"h2">) => (
      <h4 {...props} className="pt-2 text-base font-semibold text-white" />
    ),
    h3: (props: React.ComponentPropsWithoutRef<"h3">) => (
      <h5 {...props} className="pt-2 text-sm font-semibold text-white" />
    ),
    a: (props: React.ComponentPropsWithoutRef<"a">) => (
      <a
        {...props}
        className="underline decoration-white/30 underline-offset-4 hover:decoration-white"
        target="_blank"
        rel="noopener noreferrer"
      />
    ),
    code: (props: React.ComponentPropsWithoutRef<"code">) => (
      <code
        {...props}
        className="rounded bg-white/10 px-1.5 py-0.5 text-[0.85em]"
      />
    ),
    pre: (props: React.ComponentPropsWithoutRef<"pre">) => (
      <pre
        {...props}
        className="overflow-auto rounded-lg border border-white/10 bg-black/40 p-4"
      />
    ),
    li: (props: React.ComponentPropsWithoutRef<"li">) => (
      <li {...props} className="ml-4 list-disc text-zinc-200" />
    ),
    img: ({ src, alt, ...props }) => {
      const rawSrc = typeof src === "string" ? src : "";

      const resolvedSrc =
        rawSrc &&
        !rawSrc.startsWith("http://") &&
        !rawSrc.startsWith("https://") &&
        !rawSrc.startsWith("/")
          ? `${assetBaseUrl}/${rawSrc}`
          : rawSrc;

      return (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          {...props}
          src={resolvedSrc}
          alt={alt ?? ""}
          className="mt-4 max-w-full rounded-xl border border-white/10 bg-black/20"
        />
      );
    },
  };

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
                <div className="space-y-6">
                  {project.results && project.results.length > 0 ? (
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                      <h2 className="text-lg font-semibold text-white">Résultats</h2>
                      <ul className="mt-4 list-disc space-y-2 pl-5 text-sm leading-6 text-zinc-300">
                        {project.results.map((r) => (
                          <li key={r}>{r}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                    <h2 className="text-lg font-semibold text-white">Highlights</h2>
                    <ul className="mt-4 list-disc space-y-2 pl-5 text-sm leading-6 text-zinc-300">
                      {project.highlights.map((h) => (
                        <li key={h}>{h}</li>
                      ))}
                    </ul>
                  </div>

                  {project.readmePath || project.readmeUrl ? (
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <h2 className="text-lg font-semibold text-white">README</h2>
                          <p className="mt-1 text-sm text-zinc-300">
                            Documentation du projet.
                          </p>
                        </div>
                        {project.readmeUrl ? (
                          <a
                            href={project.readmeUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex h-10 items-center justify-center rounded-full border border-white/15 bg-white/5 px-4 text-sm font-semibold text-white hover:bg-white/10"
                          >
                            Open README
                          </a>
                        ) : null}
                        {githubLink ? (
                          <a
                            href={githubLink.href}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex h-10 items-center justify-center rounded-full border border-white/15 bg-white/5 px-4 text-sm font-semibold text-white hover:bg-white/10"
                          >
                            Voir sur GitHub
                          </a>
                        ) : null}
                      </div>

                      {readmeMarkdown ? (
                        <div className="mt-5 rounded-xl border border-white/10 bg-black/30 p-5">
                          <div className="space-y-4 text-sm leading-7 text-zinc-200">
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={markdownComponents}
                            >
                              {readmeMarkdown}
                            </ReactMarkdown>
                          </div>
                        </div>
                      ) : (
                        <p className="mt-5 text-sm text-zinc-300">
                          README indisponible pour le moment. Utilise le bouton “Open README”.
                        </p>
                      )}
                    </div>
                  ) : null}

                  {project.demoEmbedUrl ? (
                    <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <h2 className="text-lg font-semibold text-white">Live demo</h2>
                          <p className="mt-1 text-sm text-zinc-300">
                            Application live (si l’embed est autorisé).
                          </p>
                        </div>
                        <a
                          href={project.demoUrl ?? project.demoEmbedUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex h-10 items-center justify-center rounded-full border border-white/15 bg-white/5 px-4 text-sm font-semibold text-white hover:bg-white/10"
                        >
                          Open in new tab
                        </a>
                      </div>

                      <div className="mt-5 overflow-hidden rounded-xl border border-white/10 bg-black/30">
                        <div className="relative aspect-[16/10] w-full">
                          <iframe
                            src={project.demoEmbedUrl}
                            title={`${project.title} — Live demo`}
                            className="absolute inset-0 h-full w-full"
                            allow="clipboard-read; clipboard-write"
                            loading="lazy"
                          />
                        </div>
                      </div>
                    </div>
                  ) : null}
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
