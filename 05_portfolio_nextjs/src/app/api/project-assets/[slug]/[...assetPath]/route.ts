import { readFile } from "node:fs/promises";
import path from "node:path";

import { getProjectBySlug } from "@/data/projects";

function contentTypeForExtension(ext: string) {
  switch (ext.toLowerCase()) {
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    case ".svg":
      return "image/svg+xml";
    case ".pdf":
      return "application/pdf";
    default:
      return "application/octet-stream";
  }
}

export async function GET(
  _request: Request,
  {
    params,
  }: {
    params: Promise<{ slug: string; assetPath: string[] }>;
  }
) {
  const { slug, assetPath } = await params;

  const project = getProjectBySlug(slug);
  if (!project?.readmePath) {
    return new Response("Not found", { status: 404 });
  }

  const readmeAbsolutePath = path.resolve(process.cwd(), project.readmePath);
  const projectDir = path.dirname(readmeAbsolutePath);

  const rawRelativeAssetPath = assetPath.join("/");
  const safeRelativeAssetPath = rawRelativeAssetPath.replaceAll("\\", "/");

  const candidatePath = path.resolve(projectDir, safeRelativeAssetPath);
  if (!candidatePath.startsWith(projectDir)) {
    return new Response("Not found", { status: 404 });
  }

  try {
    const file = await readFile(candidatePath);
    const contentType = contentTypeForExtension(path.extname(candidatePath));

    return new Response(file, {
      headers: {
        "content-type": contentType,
        "cache-control": "public, max-age=3600",
      },
    });
  } catch {
    return new Response("Not found", { status: 404 });
  }
}
