import { Container } from "@/components/container";
import Image from "next/image";
import Link from "next/link";

export default function AboutPage() {
  return (
    <div className="min-h-screen">
      <section className="py-16 sm:py-20">
        <Container>
          <div className="max-w-3xl space-y-6">
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              About
            </h1>
            <div className="flex items-center gap-4">
              <div className="relative h-16 w-16 overflow-hidden rounded-full border border-white/15 bg-white/5">
                <Image
                  src="/profile/sbre-photo.jpeg"
                  alt="Soboure BELLO"
                  fill
                  className="object-cover"
                  sizes="64px"
                />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-white">Soboure BELLO</p>
                <p className="text-sm text-zinc-300">
                  Data Analyst, Data Scientist, ML Engineer & Data Engineer
                </p>
              </div>
            </div>
            <p className="text-base leading-7 text-zinc-300">
              Portfolio orienté recrutement : 4 projets end-to-end, de la conception à la
              production, couvrant Machine Learning, Deep Learning, Data Engineering et
              une application produit (dashboard).
            </p>

            <div className="grid gap-6 sm:grid-cols-2">
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                <h2 className="text-lg font-semibold text-white">Compétences</h2>
                <ul className="mt-4 list-disc space-y-2 pl-5 text-sm leading-6 text-zinc-300">
                  <li>ML: pipelines, validation, métriques</li>
                  <li>DL/NLP: entraînement, régularisation, packaging</li>
                  <li>Data Engineering: Airflow, Postgres, Docker, idempotence</li>
                  <li>Produit data: dashboards, UX, déploiement cloud</li>
                </ul>
              </div>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
                <h2 className="text-lg font-semibold text-white">Objectifs</h2>
                <ul className="mt-4 list-disc space-y-2 pl-5 text-sm leading-6 text-zinc-300">
                  <li>Montrer des preuves : UI, tables, tests, démos live</li>
                  <li>Mettre en avant une approche pro: clean code + doc</li>
                  <li>Optimiser la lisibilité pour un recruteur (2-3 minutes)</li>
                </ul>
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
              <h2 className="text-lg font-semibold text-white">Contact</h2>
              <p className="mt-3 text-sm leading-6 text-zinc-300">
                Pour échanger rapidement (opportunités, questions, collaboration) :
              </p>

              <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
                <a
                  href="https://www.linkedin.com/in/sobourebello/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex h-12 items-center justify-center gap-2 rounded-full border border-white/15 bg-white/5 px-6 text-sm font-semibold text-white hover:bg-white/10"
                >
                  <svg
                    aria-hidden="true"
                    viewBox="0 0 24 24"
                    className="h-5 w-5"
                    fill="currentColor"
                  >
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.476-.9 1.637-1.85 3.37-1.85 3.601 0 4.266 2.37 4.266 5.455v6.286zM5.337 7.433a2.063 2.063 0 1 1 0-4.126 2.063 2.063 0 0 1 0 4.126zM7.119 20.452H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.727v20.545C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.273V1.727C24 .774 23.2 0 22.222 0z" />
                  </svg>
                  LinkedIn
                </a>

                <Link
                  href="/cv/soboure-bello-cv.pdf"
                  download
                  className="inline-flex h-12 items-center justify-center gap-2 rounded-full bg-white px-6 text-sm font-semibold text-black hover:bg-zinc-200"
                >
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="opacity-90"
                  >
                    <path d="M12 3v12" />
                    <path d="m7 10 5 5 5-5" />
                    <path d="M5 21h14" />
                  </svg>
                  Download CV
                </Link>

                <a
                  href="https://github.com/soboure69"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex h-12 items-center justify-center gap-2 rounded-full border border-white/15 bg-white/5 px-6 text-sm font-semibold text-white hover:bg-white/10"
                >
                  <svg
                    aria-hidden="true"
                    viewBox="0 0 24 24"
                    className="h-5 w-5"
                    fill="currentColor"
                  >
                    <path d="M12 .5C5.73.5.75 5.6.75 12c0 5.12 3.17 9.46 7.57 10.99.55.1.76-.25.76-.55 0-.27-.01-1.15-.01-2.09-3.08.69-3.73-1.34-3.73-1.34-.5-1.33-1.22-1.68-1.22-1.68-.99-.7.07-.69.07-.69 1.1.08 1.67 1.15 1.67 1.15.98 1.74 2.58 1.24 3.2.95.1-.73.38-1.24.69-1.52-2.46-.29-5.05-1.27-5.05-5.65 0-1.25.44-2.27 1.16-3.07-.12-.3-.5-1.5.11-3.13 0 0 .95-.31 3.11 1.17a10.4 10.4 0 0 1 2.83-.39c.96 0 1.92.14 2.83.39 2.16-1.48 3.11-1.17 3.11-1.17.61 1.63.23 2.83.11 3.13.72.8 1.16 1.82 1.16 3.07 0 4.39-2.59 5.36-5.06 5.65.39.35.74 1.05.74 2.12 0 1.53-.01 2.77-.01 3.15 0 .3.21.65.77.55 4.39-1.53 7.56-5.87 7.56-10.99C23.25 5.6 18.27.5 12 .5z" />
                  </svg>
                  GitHub
                </a>
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                <a
                  href="mailto:soboure.bello@gmail.com"
                  className="inline-flex items-center justify-between rounded-2xl border border-white/10 bg-black/30 px-5 py-4 text-sm text-zinc-200 hover:bg-white/10"
                >
                  <span className="inline-flex items-center gap-2">
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 24 24"
                      className="h-5 w-5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M4 4h16v16H4z" />
                      <path d="m22 6-10 7L2 6" />
                    </svg>
                    soboure.bello@gmail.com
                  </span>
                  <span className="text-zinc-500">↗</span>
                </a>

                <a
                  href="mailto:soboure.bello@etu.univ-lyon1.fr"
                  className="inline-flex items-center justify-between rounded-2xl border border-white/10 bg-black/30 px-5 py-4 text-sm text-zinc-200 hover:bg-white/10"
                >
                  <span className="inline-flex items-center gap-2">
                    <svg
                      aria-hidden="true"
                      viewBox="0 0 24 24"
                      className="h-5 w-5"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M4 4h16v16H4z" />
                      <path d="m22 6-10 7L2 6" />
                    </svg>
                    soboure.bello@etu.univ-lyon1.fr
                  </span>
                  <span className="text-zinc-500">↗</span>
                </a>
              </div>
            </div>
          </div>
        </Container>
      </section>
    </div>
  );
}
