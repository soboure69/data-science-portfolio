import { Container } from "@/components/container";

export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-white/10 py-10">
      <Container>
        <div className="flex flex-col gap-2 text-sm text-zinc-400 sm:flex-row sm:items-center sm:justify-between">
          <p>© {year} — Data Science Portfolio</p>
          <p className="text-zinc-500">Built with Next.js + TypeScript + Tailwind</p>
        </div>
      </Container>
    </footer>
  );
}
