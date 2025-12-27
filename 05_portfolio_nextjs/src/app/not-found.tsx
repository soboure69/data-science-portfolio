import Link from "next/link";

import { Container } from "@/components/container";

export default function NotFound() {
  return (
    <div className="min-h-screen">
      <section className="py-16 sm:py-20">
        <Container>
          <div className="max-w-2xl space-y-4">
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              404 — Page introuvable
            </h1>
            <p className="text-base leading-7 text-zinc-300">
              La page demandée n’existe pas (ou l’URL est incorrecte).
            </p>
            <div className="flex flex-col gap-3 sm:flex-row">
              <Link
                href="/projects"
                className="inline-flex h-12 items-center justify-center rounded-full bg-white px-6 text-sm font-semibold text-black hover:bg-zinc-200"
              >
                Retour aux projets
              </Link>
              <Link
                href="/"
                className="inline-flex h-12 items-center justify-center rounded-full border border-white/15 bg-white/5 px-6 text-sm font-semibold text-white hover:bg-white/10"
              >
                Accueil
              </Link>
            </div>
          </div>
        </Container>
      </section>
    </div>
  );
}
