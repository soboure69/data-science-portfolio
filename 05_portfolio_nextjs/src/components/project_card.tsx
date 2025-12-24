import Image from "next/image";
import Link from "next/link";

import type { Project } from "@/data/projects";
import { Badge } from "@/components/badge";

export function ProjectCard({ project }: { project: Project }) {
  return (
    <Link
      href={`/projects/${project.slug}`}
      className="group relative rounded-2xl border border-white/10 bg-white/5 p-6 transition hover:bg-white/10"
    >
      {project.coverImage ? (
        <div className="mb-5 overflow-hidden rounded-xl border border-white/10 bg-black/30">
          <div className="relative aspect-[16/9] w-full">
            <Image
              src={project.coverImage.src}
              alt={project.coverImage.alt}
              fill
              className="object-cover"
              sizes="(max-width: 768px) 100vw, 520px"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/55 via-black/10 to-transparent" />
          </div>
        </div>
      ) : null}

      <div className="flex items-start justify-between gap-4">
        <div className="space-y-2">
          <p className="text-xs font-medium text-zinc-400">{project.year}</p>
          <h3 className="text-lg font-semibold text-white group-hover:text-white">
            {project.title}
          </h3>
          <p className="text-sm leading-6 text-zinc-300">{project.subtitle}</p>
        </div>
        <div className="hidden shrink-0 sm:block">
          <span className="inline-flex items-center rounded-full border border-white/10 bg-black/40 px-3 py-1 text-xs font-medium text-zinc-200">
            View
          </span>
        </div>
      </div>

      <div className="mt-5 flex flex-wrap gap-2">
        {project.stack.slice(0, 6).map((t) => (
          <Badge key={t}>{t}</Badge>
        ))}
      </div>

      <div className="mt-6 grid grid-cols-2 gap-3">
        {project.metrics.slice(0, 2).map((m) => (
          <div
            key={m.label}
            className="rounded-xl border border-white/10 bg-black/30 p-3"
          >
            <p className="text-xs text-zinc-400">{m.label}</p>
            <p className="text-sm font-medium text-zinc-100">{m.value}</p>
          </div>
        ))}
      </div>
    </Link>
  );
}
