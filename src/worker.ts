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
  modelId: string;
  precision: 'int8' | 'int4';
  calibrationData?: string;
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

    const compiledModel = await compileModel(model, data);
    
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
    await env.COMPILER_CACHE.put(cacheKey, JSON.stringify({
      ...data,
      status: 'failed',
      error: error.message,
      failedAt: Date.now()
    }), { expirationTtl: 3600 });
  }
}

async function compileModel(model: R2Object, options: CompileRequest): Promise<ArrayBuffer> {
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

  return result.compiledModel;
}

async function handleQuantize(request: Request, env: Env): Promise<Response> {
  try {
    const data: QuantizeRequest = await request.json();
    
    if (!data.modelId || !data.precision) {
      return new Response(JSON.stringify({ error: "Missing required fields" }), {
        status: 400,
        headers: { "Content-Type": "application/json", ...CORS_HEADERS }
      });
    }

    const model = await env.MODEL_STORE.get(data.modelId);
    if (!model) {
      return new Response(JSON.stringify({ error: "Model not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json", ...CORS_HEADERS }
      });
    }

    const quantizationOptions = {
      precision: data.precision,
      calibrationData: data.calibrationData,
      symmetric: true,
      perChannel: true
    };

    const modelData = await model.arrayBuffer();
    const quantized = await env.AI.run("@cf/quantization", {
      model: new Uint8Array(modelData),
      options: quantizationOptions
    });

    const quantizedKey = `quantized/${data.modelId}_${data.precision}`;
    await env.MODEL_STORE.put(quantizedKey, quantized.model);

    return new Response(JSON.stringify({
      modelId: data.modelId,
      precision: data.precision,
      sizeReduction: `${((1 - quantized.model.byteLength / modelData.byteLength) * 100).toFixed(1)}%`,
      downloadUrl: `/api/download/${quantizedKey}`
    }), {
      headers: { "Content-Type": "application/json", ...CORS_HEADERS }
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: "Quantization failed" }), {
      status: 500,
      headers: { "Content-Type": "application/json", ...CORS_HEADERS }
    });
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
