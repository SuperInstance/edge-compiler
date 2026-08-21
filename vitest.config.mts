import { defineConfig } from "vitest/config";
import { cloudflareTest } from "@cloudflare/vitest-pool-workers";

export default defineConfig({
  // Run tests inside the real workerd runtime (via @cloudflare/vitest-pool-workers)
  // with in-memory stubs for KV/R2. No real Cloudflare account needed.
  // Uses wrangler.test.toml (no [ai] binding) so no remote Workers AI connection
  // is opened during tests; /api/quantize does not use env.AI.
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.test.toml" },
      miniflare: {
        kvStores: ["COMPILER_CACHE"],
        r2Buckets: ["MODEL_STORE"],
      },
    }),
  ],
});
