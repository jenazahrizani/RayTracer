import { defineConfig } from "astro/config";

const repository =
  process.env.GITHUB_REPOSITORY?.split("/")[1] ||
  "RayTracer";

const owner =
  process.env.GITHUB_REPOSITORY_OWNER ||
  process.env.GITHUB_REPOSITORY?.split("/")[0] ||
  "";

const isGitHubActions =
  process.env.GITHUB_ACTIONS === "true";

const base =
  isGitHubActions
    ? `/${repository}`
    : "/";

const site =
  isGitHubActions && owner
    ? `https://${owner}.github.io${base}`
    : undefined;

export default defineConfig({
  output: "static",

  base,

  site,

  trailingSlash: "never",

  build: {
    format: "directory",
  },

  vite: {
    build: {
      target: "es2022",
    },
  },
});