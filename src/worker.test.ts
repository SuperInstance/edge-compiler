import { describe, it, expect } from "vitest";
import { SELF, env } from "cloudflare:test";
import type { Env } from "./worker";
import { decodeQuantizedBlob } from "./quantize";

// `env` from cloudflare:test is typed as the global Cloudflare.Env; cast to the
// Worker's own Env for typed access to bindings in tests.
const testEnv = env as unknown as Env;

interface QuantizeResponseBody {
  precision?: string;
  method?: string;
  elementCount?: number;
  originalBytes?: number;
  quantizedBytes?: number;
  sizeReduction?: string;
  scale?: number;
  downloadUrl?: string;
  modelId?: string;
  error?: string;
}

/**
 * End-to-end integration test: drives the real Worker's HTTP layer (running in
 * the workerd runtime via @cloudflare/vitest-pool-workers) with an in-memory R2
 * stub. Proves the Worker actually runs the real INT8 quantization and returns
 * real, hand-computed numbers — not just that the code compiles.
 */

// Same hand-computed reference as the unit test.
const INPUT_VALUES = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.125];
const EXPECTED_INT8 = [127, -127, 64, -63, 32, -32, 0, 16];

async function postQuantize(body: unknown): Promise<Response> {
  return SELF.fetch("http://localhost/api/quantize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function downloadBlob(downloadUrl: string | undefined): Promise<Uint8Array> {
  expect(downloadUrl).toBeDefined();
  const res = await SELF.fetch(`http://localhost${downloadUrl}`);
  expect(res.status).toBe(200);
  return new Uint8Array(await res.arrayBuffer());
}

describe("POST /api/quantize (real INT8 pass via inline values)", () => {
  it("returns the hand-computed size reduction, scale and byte counts", async () => {
    const res = await postQuantize({ precision: "int8", values: INPUT_VALUES });
    expect(res.status).toBe(200);
    const body = (await res.json()) as QuantizeResponseBody;

    expect(body.precision).toBe("int8");
    expect(body.method).toBe("symmetric-per-tensor-int8");
    expect(body.elementCount).toBe(8);
    // 8 floats * 4 bytes = 32 in; blob = 8-byte header + 8-byte payload = 16 out.
    expect(body.originalBytes).toBe(32);
    expect(body.quantizedBytes).toBe(16);
    // 1 - 16/32 = 0.5 => "50.0%"
    expect(body.sizeReduction).toBe("50.0%");
    expect(body.scale).toBeCloseTo(1 / 127, 6);
    expect(body.downloadUrl).toMatch(/^\/api\/download\/quantized\//);
  });

  it("stores a real, downloadable, decodable quantized blob", async () => {
    const res = await postQuantize({ precision: "int8", values: INPUT_VALUES });
    const body = (await res.json()) as QuantizeResponseBody;

    const blob = await downloadBlob(body.downloadUrl);
    expect(blob.byteLength).toBe(16);
    const decoded = decodeQuantizedBlob(blob);
    expect(decoded.magic).toBe("QNT1");
    expect(Array.from(decoded.quantized)).toEqual(EXPECTED_INT8);
    expect(decoded.scale).toBeCloseTo(1 / 127, 5);
  });
});

describe("POST /api/quantize (base64 raw float32 buffer input)", () => {
  it("quantizes a raw FP32 buffer and returns the same real numbers", async () => {
    const f32 = new Float32Array(INPUT_VALUES);
    const bytes = new Uint8Array(f32.buffer);
    // btoa on the raw bytes
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    const data = btoa(bin);

    const res = await postQuantize({ precision: "int8", data });
    expect(res.status).toBe(200);
    const body = (await res.json()) as QuantizeResponseBody;
    expect(body.elementCount).toBe(8);
    expect(body.originalBytes).toBe(32);
    expect(body.quantizedBytes).toBe(16);
    expect(body.sizeReduction).toBe("50.0%");

    const blob = await downloadBlob(body.downloadUrl);
    expect(Array.from(decodeQuantizedBlob(blob).quantized)).toEqual(EXPECTED_INT8);
  });
});

describe("POST /api/quantize (R2 modelId input)", () => {
  it("reads a float32 buffer from R2 and quantizes it", async () => {
    const f32 = new Float32Array(INPUT_VALUES);
    await testEnv.MODEL_STORE.put("r2model", f32.buffer);

    const res = await postQuantize({ precision: "int8", modelId: "r2model" });
    expect(res.status).toBe(200);
    const body = (await res.json()) as QuantizeResponseBody;
    expect(body.modelId).toBe("r2model");
    expect(body.sizeReduction).toBe("50.0%");
    expect(body.downloadUrl).toBe("/api/download/quantized/r2model_int8");

    const blob = await downloadBlob(body.downloadUrl);
    expect(Array.from(decodeQuantizedBlob(blob).quantized)).toEqual(EXPECTED_INT8);
  });

  it("returns 404 for an unknown modelId", async () => {
    const res = await postQuantize({ precision: "int8", modelId: "does-not-exist" });
    expect(res.status).toBe(404);
  });
});

describe("POST /api/quantize (validation & honesty)", () => {
  it("returns 501 for int4 (not implemented, not faked)", async () => {
    const res = await postQuantize({ precision: "int4", values: INPUT_VALUES });
    expect(res.status).toBe(501);
    const body = (await res.json()) as QuantizeResponseBody;
    expect(body.error).toMatch(/int4 quantization is not implemented/i);
  });

  it("returns 400 when no input is provided", async () => {
    const res = await postQuantize({ precision: "int8" });
    expect(res.status).toBe(400);
  });

  it("returns 400 when precision is missing", async () => {
    const res = await postQuantize({ values: INPUT_VALUES });
    expect(res.status).toBe(400);
  });

  it("returns 400 for a malformed float32 buffer (length not a multiple of 4)", async () => {
    // 5 bytes -> not a valid float32 buffer
    let bin = "abcde";
    const res = await postQuantize({ precision: "int8", data: btoa(bin) });
    expect(res.status).toBe(400);
  });
});

describe("POST /api/compile (validation)", () => {
  async function postCompile(body: unknown): Promise<Response> {
    return SELF.fetch("http://localhost/api/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  it("returns 400 when required fields are missing", async () => {
    const res = await postCompile({ modelId: "m" });
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error?: string };
    expect(body.error).toMatch(/missing required fields/i);
  });

  it("returns 400 for an unsupported hardware target", async () => {
    const res = await postCompile({
      modelId: "m",
      target: "onnx",
      hardware: "nvidia-a100",
      precision: "fp32",
    });
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error?: string };
    expect(body.error).toMatch(/unsupported hardware/i);
  });

  it("returns 400 when the precision is not supported for the hardware", async () => {
    // raspberry-pi-4 only supports int8.
    const res = await postCompile({
      modelId: "m",
      target: "tflite",
      hardware: "raspberry-pi-4",
      precision: "fp32",
    });
    expect(res.status).toBe(400);
    const body = (await res.json()) as { error?: string };
    expect(body.error).toMatch(/not supported/i);
  });

  it("returns 202 with a queued job shape", async () => {
    const res = await postCompile({
      modelId: "m",
      target: "onnx",
      hardware: "nvidia-t4",
      precision: "fp32",
    });
    expect(res.status).toBe(202);
    const body = (await res.json()) as {
      jobId?: string;
      status?: string;
      estimatedTime?: number;
    };
    expect(body.jobId).toMatch(/^[0-9a-f-]{36}$/i);
    expect(body.status).toBe("queued");
    expect(body.estimatedTime).toBe(120);
  });
});

describe("POST /api/compile (background job scheduling via ctx.waitUntil)", () => {
  async function postCompile(body: unknown): Promise<Response> {
    return SELF.fetch("http://localhost/api/compile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  // Reads the raw KV record for a compile job so we can assert on the status
  // transitions written by the background task.
  async function readCompileJob(
    jobId: string,
  ): Promise<{ status?: string; error?: string } | null> {
    const raw = await testEnv.COMPILER_CACHE.get(`compile:${jobId}`);
    if (raw === null) return null;
    return JSON.parse(raw) as { status?: string; error?: string };
  }

  // ctx.waitUntil'd tasks settle asynchronously after the fetch response is
  // returned; poll KV until the background job reaches a terminal status.
  async function waitForTerminalStatus(
    jobId: string,
    timeoutMs = 2000,
  ): Promise<{ status?: string; error?: string }> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const job = await readCompileJob(jobId);
      if (job && (job.status === "completed" || job.status === "failed")) {
        return job;
      }
      await new Promise((r) => setTimeout(r, 10));
    }
    throw new Error(`timed out waiting for compile job ${jobId} to settle`);
  }

  it("actually runs the background job: KV transitions from queued to a terminal status", async () => {
    // modelId "missing-model" is intentionally not in R2, so the background
    // job deterministically fails at the lookup step ("Model not found")
    // without ever touching env.AI (which is unbound in the test runtime).
    const res = await postCompile({
      modelId: "missing-model",
      target: "onnx",
      hardware: "nvidia-t4",
      precision: "fp32",
    });
    expect(res.status).toBe(202);
    const { jobId } = (await res.json()) as { jobId: string };

    // With the old setTimeout()-based scheduling, the background task was not
    // registered with the execution context and the job would stay "queued"
    // forever. With ctx.waitUntil(), it settles to "failed" here.
    const job = await waitForTerminalStatus(jobId);
    expect(job.status).toBe("failed");
    expect(job.error).toMatch(/model not found/i);
  });
});
