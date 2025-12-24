"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import type { Project } from "@/data/projects";

export function HeroCarousel({ projects }: { projects: Project[] }) {
  const items = useMemo(
    () => projects.filter((p) => p.coverImage).slice(0, 6),
    [projects]
  );

  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduceMotion(Boolean(media.matches));
    update();

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", update);
      return () => media.removeEventListener("change", update);
    }

    media.addListener(update);
    return () => media.removeListener(update);
  }, []);

  useEffect(() => {
    if (items.length <= 1) return;
    if (reduceMotion) return;
    if (paused) return;

    const id = window.setInterval(() => {
      setActive((i) => (i + 1) % items.length);
    }, 4500);

    return () => window.clearInterval(id);
  }, [items.length, paused, reduceMotion]);

  if (items.length === 0) return null;

  const project = items[active];

  return (
    <div
      className="overflow-hidden rounded-3xl border border-white/10 bg-white/5"
      onMouseEnter={() => setPaused(true)}
      onMouseLeave={() => setPaused(false)}
      onFocusCapture={() => setPaused(true)}
      onBlurCapture={() => setPaused(false)}
    >
      <div className="relative aspect-[16/9] w-full">
        {project.coverImage ? (
          <Image
            src={project.coverImage.src}
            alt={project.coverImage.alt}
            fill
            className="object-cover"
            sizes="(max-width: 1024px) 100vw, 640px"
            priority
          />
        ) : null}
        <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/20 to-transparent" />

        <div className="absolute bottom-0 left-0 right-0 p-5 sm:p-6">
          <div className="flex flex-col gap-2">
            <p className="text-xs font-medium text-zinc-300">Featured project</p>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <p className="text-lg font-semibold text-white">{project.title}</p>
                <p className="mt-1 text-sm text-zinc-300">{project.subtitle}</p>
              </div>
              <Link
                href={`/projects/${project.slug}`}
                className="inline-flex h-11 items-center justify-center rounded-full bg-white px-5 text-sm font-semibold text-black hover:bg-zinc-200"
              >
                Voir le détail
              </Link>
            </div>

            <div className="mt-3 flex items-center gap-2">
              {items.map((p, idx) => (
                <button
                  key={p.slug}
                  type="button"
                  aria-label={`Go to ${p.title}`}
                  onClick={() => setActive(idx)}
                  className={`h-2.5 w-2.5 rounded-full transition ${
                    idx === active ? "bg-white" : "bg-white/25 hover:bg-white/50"
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
