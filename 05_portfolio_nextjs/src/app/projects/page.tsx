import { Container } from "@/components/container";
import { ProjectCard } from "@/components/project_card";
import { projects } from "@/data/projects";

export default function ProjectsPage() {
  return (
    <div className="min-h-screen">
      <section className="py-16 sm:py-20">
        <Container>
          <div className="max-w-2xl space-y-4">
            <h1 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              Projets
            </h1>
            <p className="text-base leading-7 text-zinc-300">
              Une sélection de 4 projets couvrant ML, Deep Learning, Data Engineering et
              produit data (dashboard + déploiement).
            </p>
          </div>

          <div className="mt-10 grid gap-6 md:grid-cols-2">
            {projects.map((p) => (
              <ProjectCard key={p.slug} project={p} />
            ))}
          </div>
        </Container>
      </section>
    </div>
  );
}
