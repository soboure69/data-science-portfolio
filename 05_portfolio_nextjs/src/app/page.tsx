import { HeroCarousel } from "@/components/hero_carousel";
import { ProjectCard } from "@/components/project_card";
import { projects } from "@/data/projects";
import Image from "next/image";
import Link from "next/link";

export default function Home() {
  const featuredProjects =
    projects.length >= 2
      ? [projects[0], projects[projects.length - 1]]
      : projects;

  return (
    <div>
      <section className="py-16 sm:py-24">
        <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="grid gap-10 lg:grid-cols-2 lg:items-center">
            <div className="space-y-6">
              <p className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-xs font-medium text-zinc-200">
                Portfolio end-to-end — conception → production
              </p>
              <div className="flex items-center gap-4">
                <div className="relative h-16 w-16 overflow-hidden rounded-full border border-white/15 bg-white/5">
                  <Image
                    src="/profile/sbre-photo.jpeg"
                    alt="Soboure BELLO"
                    fill
                    className="object-cover"
                    sizes="64px"
                    priority
                  />
                </div>
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-white">Soboure BELLO</p>
                  <p className="text-sm text-zinc-300">Data Analyst • Data Scientist • ML Engineer • Data Engineer</p>
                </div>
              </div>
              <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-5xl">
                Soboure BELLO
                <span className="block text-zinc-300">
                  Data Analyst, Data Scientist, ML Engineer & Data Engineer
                </span>
              </h1>
              <p className="max-w-xl text-base leading-7 text-zinc-300">
                Portfolio end-to-end : Machine Learning, Deep Learning (NLP), Data Engineering (ETL)
                et une application produit (dashboard + déploiement cloud). Conçu pour une lecture
                recruteur en 2–3 minutes.
              </p>
              <div className="flex flex-col gap-3 sm:flex-row">
                <Link
                  href="/projects"
                  className="inline-flex h-12 items-center justify-center rounded-full bg-white px-6 text-sm font-semibold text-black hover:bg-zinc-200"
                >
                  Voir les projets
                </Link>
                <Link
                  href="/about"
                  className="inline-flex h-12 items-center justify-center rounded-full border border-white/15 bg-white/5 px-6 text-sm font-semibold text-white hover:bg-white/10"
                >
                  À propos
                </Link>
              </div>
            </div>

            <div className="space-y-6">
              <HeroCarousel projects={projects} />

              <div className="rounded-3xl border border-white/10 bg-gradient-to-b from-white/10 to-white/5 p-6">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="rounded-2xl border border-white/10 bg-black/30 p-5">
                    <p className="text-xs text-zinc-400">Evidence</p>
                    <p className="mt-2 text-sm font-semibold text-white">Démos live</p>
                    <p className="mt-1 text-sm text-zinc-300">Apps déployées + screenshots + GIFs</p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-black/30 p-5">
                    <p className="text-xs text-zinc-400">Engineering</p>
                    <p className="mt-2 text-sm font-semibold text-white">Pipelines & tests</p>
                    <p className="mt-1 text-sm text-zinc-300">ETL, idempotence, sanity checks</p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-black/30 p-5">
                    <p className="text-xs text-zinc-400">ML</p>
                    <p className="mt-2 text-sm font-semibold text-white">Évaluation solide</p>
                    <p className="mt-1 text-sm text-zinc-300">CV, métriques, anti-leakage</p>
                  </div>
                  <div className="rounded-2xl border border-white/10 bg-black/30 p-5">
                    <p className="text-xs text-zinc-400">Product</p>
                    <p className="mt-2 text-sm font-semibold text-white">UX & story</p>
                    <p className="mt-1 text-sm text-zinc-300">Filtres, cas limites, business case</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-10 sm:py-16">
        <div className="mx-auto w-full max-w-6xl px-4 sm:px-6 lg:px-8">
          <div className="flex items-end justify-between gap-6">
            <div className="space-y-2">
              <h2 className="text-2xl font-semibold tracking-tight text-white">Projets</h2>
              <p className="text-sm text-zinc-300">Accès rapide aux pages de détail.</p>
            </div>
            <Link
              href="/projects"
              className="hidden rounded-full border border-white/15 bg-white/5 px-4 py-2 text-sm font-medium text-white hover:bg-white/10 sm:inline-flex"
            >
              Tout voir
            </Link>
          </div>

          <div className="mt-8 grid gap-6 md:grid-cols-2">
            {featuredProjects.map((p) => (
              <ProjectCard key={p.slug} project={p} />
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
