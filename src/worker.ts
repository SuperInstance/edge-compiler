import {
  quantizeFloat32ToInt8,
  encodeQuantizedBlob,
  sizeReductionFraction,
} from "./quantize";

export interface Env {
  COMPILER_CACHE: KVNamespace;
  MODEL_STORE: R2Bucket;
  AI: Ai;
}

interface CompileRequest {
  modelId: string;
  target: 'onnx' | 'tensorrt' | 'tflite';
  hardware: string;
  batchSize?: number;
  precision: 'fp32' | 'fp16' | 'int8' | 'int4';
}

interface QuantizeRequest {
  /** R2 key of a pre-uploaded raw float32 buffer to quantize. */
  modelId?: string;
  precision: 'int8' | 'int4';
  /** Inline float values to quantize (alternative to modelId/data). */
  values?: number[];
  /** Base64 of a raw float32 byte buffer to quantize (alternative to modelId/values). */
  data?: string;
  /** Accepted for backward compatibility; ignored by the real int8 pass. */
  calibrationData?: string;
}

interface QuantizeResponse {
  modelId?: string;
  precision: 'int8' | 'int4';
  method: string;
  scale: number;
  elementCount: number;
  originalBytes: number;
  quantizedBytes: number;
  sizeReduction: string;
  downloadUrl: string;
}

interface CompileResponse {
  jobId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  estimatedTime?: number;
  downloadUrl?: string;
}

interface HardwareTarget {
  id: string;
  name: string;
  supportedPrecisions: string[];
  maxBatchSize: number;
  memoryLimit: number;
}

const HARDWARE_TARGETS: HardwareTarget[] = [
  {
    id: "nvidia-t4",
    name: "NVIDIA T4 GPU",
    supportedPrecisions: ["fp32", "fp16", "int8"],
    maxBatchSize: 64,
    memoryLimit: 16 * 1024 * 1024 * 1024
  },
  {
    id: "jetson-nano",
    name: "NVIDIA Jetson Nano",
    supportedPrecisions: ["fp16", "int8"],
    maxBatchSize: 8,
    memoryLimit: 4 * 1024 * 1024 * 1024
  },
  {
    id: "raspberry-pi-4",
    name: "Raspberry Pi 4",
    supportedPrecisions: ["int8"],
    maxBatchSize: 4,
    memoryLimit: 8 * 1024 * 1024 * 1024
  },
  {
    id: "apple-m1",
    name: "Apple M1",
    supportedPrecisions: ["fp32", "fp16", "int8"],
    maxBatchSize: 32,
    memoryLimit: 16 * 1024 * 1024 * 1024
  }
];

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Max-Age": "86400"
};

function jsonError(status: number, message: string): Response {
  return new Response(JSON.stringify({ error: message }), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

async function handleCompile(request: Request, env: Env): Promise<Response> {
  try {
    const data: CompileRequest = await request.json();
    
    if (!data.modelId || !data.target || !data.hardware || !data.precision) {
      return new Response(JSON.stringify({ error: "Missing required fields" }), {
        status: 400,
        headers: { "Content-Type": "application/json", ...CORS_HEADERS }
      });
    }

    const hardware = HARDWARE_TARGETS.find(h => h.id === data.hardware);
    if (!hardware) {
      return new Response(JSON.stringify({ error: "Unsupported hardware target" }), {
        status: 400,
        headers: { "Content-Type": "application/json", ...CORS_HEADERS }
      });
    }

    if (!hardware.supportedPrecisions.includes(data.precision)) {
      return new Response(JSON.stringify({ 
        error: `Precision ${data.precision} not supported for ${hardware.name}` 
      }), {
        status: 400,
        headers: { "Content-Type": "application/json", ...CORS_HEADERS }
      });
    }

    const jobId = crypto.randomUUID();
    const cacheKey = `compile:${jobId}`;
    
    const response: CompileResponse = {
      jobId,
      status: 'queued',
      estimatedTime: 120
    };

    await env.COMPILER_CACHE.put(cacheKey, JSON.stringify({
      ...data,
      status: 'queued',
      createdAt: Date.now()
    }), { expirationTtl: 3600 });

    setTimeout(() => processCompilationJob(jobId, data, env), 100);

    return new Response(JSON.stringify(response), {
      status: 202,
      headers: { "Content-Type": "application/json", ...CORS_HEADERS }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Invalid request" }), {
      status: 400,
      headers: { "Content-Type": "application/json", ...CORS_HEADERS }
    });
  }
}

async function processCompilationJob(jobId: string, data: CompileRequest, env: Env) {
  const cacheKey = `compile:${jobId}`;
  
  try {
    await env.COMPILER_CACHE.put(cacheKey, JSON.stringify({
      ...data,
      status: 'processing',
      updatedAt: Date.now()
    }), { expirationTtl: 3600 });

    const model = await env.MODEL_STORE.get(data.modelId);
    if (!model) {
      throw new Error("Model not found");
    }

    const compiledModel = await compileModel(model, data, env);
    
    const outputKey = `compiled/${jobId}/${data.modelId}.${data.target}`;
    await env.MODEL_STORE.put(outputKey, compiledModel);
    
    const downloadUrl = `/api/download/${jobId}`;
    
    await env.COMPILER_CACHE.put(cacheKey, JSON.stringify({
      ...data,
      status: 'completed',
      downloadUrl,
      completedAt: Date.now()
    }), { expirationTtl: 86400 });
    
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    await env.COMPILER_CACHE.put(cacheKey, JSON.stringify({
      ...data,
      status: 'failed',
      error: errorMessage,
      failedAt: Date.now()
    }), { expirationTtl: 3600 });
  }
}

async function compileModel(model: R2ObjectBody, options: CompileRequest, env: Env): Promise<ArrayBuffer> {
  const modelData = await model.arrayBuffer();
  
  const compilationOptions = {
    target: options.target,
    hardware: options.hardware,
    precision: options.precision,
    batchSize: options.batchSize || 1,
    optimize: true,
    fuseOperations: true,
    memoryOptimization: true
  };

  const result = await env.AI.run("@cf/onnx", {
    model: new Uint8Array(modelData),
    options: compilationOptions
  });

  const compiled = result as { compiledModel: ArrayBuffer };
  return compiled.compiledModel;
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function bytesToFloat32(bytes: Uint8Array): Float32Array {
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(
      `float32 buffer length (${bytes.byteLength}) is not a multiple of 4`,
    );
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

async function handleQuantize(request: Request, env: Env): Promise<Response> {
  try {
    const data: QuantizeRequest = await request.json();

    if (!data.precision) {
      return jsonError(400, "Missing required field: precision");
    }
    if (data.precision !== "int8" && data.precision !== "int4") {
      return jsonError(400, `Unsupported precision: ${data.precision}`);
    }
    if (data.precision === "int4") {
      return jsonError(
        501,
        "int4 quantization is not implemented (only int8 is real).",
      );
    }

    let float32: Float32Array;
    try {
      if (data.values && data.values.length > 0) {
        float32 = new Float32Array(data.values);
      } else if (data.data) {
        float32 = bytesToFloat32(base64ToBytes(data.data));
      } else if (data.modelId) {
        const obj = await env.MODEL_STORE.get(data.modelId);
        if (!obj) return jsonError(404, "Model not found");
        float32 = bytesToFloat32(new Uint8Array(await obj.arrayBuffer()));
      } else {
        return jsonError(
          400,
          "No input provided: supply one of 'values', 'data', or 'modelId'",
        );
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return jsonError(400, `Invalid input buffer: ${msg}`);
    }

    const result = quantizeFloat32ToInt8(float32);
    const blob = encodeQuantizedBlob(result);

    const quantizedKey = data.modelId
      ? `quantized/${data.modelId}_int8`
      : `quantized/buffer_int8_${crypto.randomUUID()}`;
    await env.MODEL_STORE.put(quantizedKey, blob);

    const response: QuantizeResponse = {
      modelId: data.modelId,
      precision: "int8",
      method: "symmetric-per-tensor-int8",
      scale: result.scale,
      elementCount: float32.length,
      originalBytes: result.originalBytes,
      quantizedBytes: blob.byteLength,
      sizeReduction: `${(sizeReductionFraction(result.originalBytes, blob.byteLength) * 100).toFixed(1)}%`,
      downloadUrl: `/api/download/${quantizedKey}`,
    };

    return new Response(JSON.stringify(response), {
      headers: { "Content-Type": "application/json", ...CORS_HEADERS },
    });
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    return jsonError(500, `Quantization failed: ${msg}`);
  }
}

async function handleTargets(): Promise<Response> {
  return new Response(JSON.stringify({
    targets: HARDWARE_TARGETS.map(t => ({
      id: t.id,
      name: t.name,
      capabilities: {
        supportedPrecisions: t.supportedPrecisions,
        maxBatchSize: t.maxBatchSize,
        memoryLimit: t.memoryLimit
      }
    }))
  }), {
    headers: { 
      "Content-Type": "application/json",
      ...CORS_HEADERS
    }
  });
}

async function handleHealth(): Promise<Response> {
  return new Response(JSON.stringify({
    status: "healthy",
    timestamp: new Date().toISOString(),
    version: "1.0.0"
  }), {
    headers: { 
      "Content-Type": "application/json",
      "Cache-Control": "no-cache"
    }
  });
}

async function handleOptions(): Promise<Response> {
  return new Response(null, {
    headers: CORS_HEADERS
  });
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;

    if (request.method === "OPTIONS") {
      return handleOptions();
    }

    if (path === "/health" || path === "/api/health") {
      return handleHealth();
    }

    if (path === "/api/targets" && request.method === "GET") {
      return handleTargets();
    }

    if (path === "/api/compile" && request.method === "POST") {
      return handleCompile(request, env);
    }

    if (path === "/api/quantize" && request.method === "POST") {
      return handleQuantize(request, env);
    }

    if (path.startsWith("/api/download/") && request.method === "GET") {
      const key = path.replace("/api/download/", "");
      const object = await env.MODEL_STORE.get(key);
      
      if (!object) {
        return new Response("Not found", { status: 404 });
      }

      return new Response(object.body, {
        headers: {
          "Content-Type": "application/octet-stream",
          "Content-Disposition": `attachment; filename="${key.split('/').pop()}"`
        }
      });
    }

    return new Response(JSON.stringify({
      name: "Edge Compiler",
      version: "1.0.0",
      endpoints: [
        "POST /api/compile",
        "GET /api/targets",
        "POST /api/quantize",
        "GET /api/download/{id}",
        "GET /health"
      ]
    }), {
      status: 404,
      headers: { "Content-Type": "application/json", ...CORS_HEADERS }
    });
  }
};
