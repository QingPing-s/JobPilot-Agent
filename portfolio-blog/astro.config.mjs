import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";

export default defineConfig({
  site: "https://QingPing-s.github.io",
  base: "/JobPilot-Agent",
  integrations: [sitemap()],
});
