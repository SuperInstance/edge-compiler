# edge-compiler

Compile and optimize machine learning models for specific hardware targets using Cloudflare Workers Edge Compiler.

## Endpoints

### `POST /api/compile`
Submit a compilation job.

**Request body**
```json
{
  "modelId": "string (required)",
  "target": "onnx | tensorrt | tflite (required)",
  "hardware": "nvidia-t4 | jetson-nano | raspberry-pi-4 | apple-m1 (required)",
  "batchSize": "number (optional, default 1)",
  "precision": "fp32 | fp16 | int8 | int4 (required)"
}
```

**Response (202 Accepted)**
```json
{
  "jobId": "uuid",
  "status": "queued",
  "estimatedTime": 120
}
```

The job is tracked asynchronously via KV. Retrieve the compiled model later with `GET /api/download/<jobId>`.

### `POST /api/quantize`
Quantize a raw FP32 tensor buffer to INT8. This is a **real, working**
quantization pass implemented directly in the Worker (not an external model).

**Request body** — supply exactly one input source (`values`, `data`, or
`modelId`):
```json
{
  "precision": "int8",
  "values": [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.125]
}
```
| Field         | Type     | Description                                                            |
|---------------|----------|------------------------------------------------------------------------|
| `precision`   | string   | `"int8"` (the only precision actually implemented; `"int4"` returns `501`) |
| `values`      | number[] | Inline float values to quantize                                        |
| `data`        | string   | Base64 of a raw little-endian float32 byte buffer                      |
| `modelId`     | string   | R2 key of a pre-uploaded raw float32 buffer                            |
| `calibrationData` | string | Accepted for backward compatibility; ignored by the real int8 pass |

The algorithm is **symmetric per-tensor INT8 linear quantization**:
`absmax = max(|x_i|)`, `scale = absmax / 127`, `q_i = clamp(round(x_i / scale), -128, 127)`.
Rounding follows `Math.round` semantics (halves toward `+Infinity`).

**Response (200 OK)**
```json
{
  "precision": "int8",
  "method": "symmetric-per-tensor-int8",
  "scale": 0.007874015748031496,
  "elementCount": 8,
  "originalBytes": 32,
  "quantizedBytes": 16,
  "sizeReduction": "50.0%",
  "downloadUrl": "/api/download/quantized/buffer_int8_<uuid>"
}
```
(`modelId` is also echoed back when the `modelId` input source was used.)
`quantizedBytes` and `sizeReduction` reflect the **actual stored blob** (an
8-byte header — magic `QNT1` + float32 `scale` — followed by the int8 payload),
so the reported reduction is honest and downloadable. `GET` the `downloadUrl`
to fetch the blob; decode it with `decodeQuantizedBlob()` from `src/quantize.ts`
and dequantize with `dequantizeInt8ToFloat32()`.

### `GET /api/targets`
List available hardware targets and their capabilities.

**Response (200 OK)**
```json
{
  "targets": [
    {
      "id": "nvidia-t4",
      "name": "NVIDIA T4 GPU",
      "capabilities": {
        "supportedPrecisions": ["fp32","fp16","int8"],
        "maxBatchSize": 64,
        "memoryLimit": 17179869184
      }
    },
    ...
  ]
}
```

### `GET /api/download/{key}`
Download a compiled or quantized model by its key (the `jobId` for compile jobs, or the quantized key for quantize jobs).

**Response** – binary octet-stream with `Content-Disposition: attachment`.

### `GET /health`
Health check.

**Response (200 OK)**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-01T00:00:00.000Z",
  "version": "1.0.0"
}
```

## 📦 Required Bindings

The worker expects the following Cloudflare resources:

| Binding           | Type        | Purpose                                         |
|-------------------|-------------|-------------------------------------------------|
| `COMPILER_CACHE`  | KV Namespace| Track compile job status                        |
| `MODEL_STORE`     | R2 Bucket   | Store input & output models / quantized blobs   |
| `AI`              | Workers AI  | Used only by the 🔮 compile path (see below)     |

These are declared as placeholders in `wrangler.toml` — replace the
namespace/bucket IDs with your own before deploying:

```bash
npx wrangler kv namespace create COMPILER_CACHE
npx wrangler r2 bucket create model-store
# Then update wrangler.toml with the generated IDs.
```

## ✅ / 🔮 Status of the Model-Transformation Step

This section is kept honest and up to date. It was re-verified against
Cloudflare's **live** Workers AI model catalog
(`https://developers.cloudflare.com/workers-ai/models/`, 81 models as of
2026-07-10) and the Workers AI documentation index (`llms.txt`).

### What was found in the real catalog
Workers AI exposes **inference models only** (text generation, image, audio,
embeddings, classification, translation, ASR, object detection). The only
quantization-adjacent entries are **pre-quantized inference models** you *run*
— e.g. `@cf/meta/llama-3.1-8b-instruct-awq` (int4),
`@cf/meta/llama-3.1-8b-instruct-fp8`, `@cf/meta/llama-3.3-70b-instruct-fp8-fast`,
`@cf/meta/llama-2-7b-chat-int8`, `@cf/huggingface/distilbert-sst-2-int8`. There
is **no** model that *transforms* an input model, and **no** `@cf/onnx` or
`@cf/quantization` model ID. No ONNX-compilation or quantization-as-a-service
feature exists in Workers AI.

### ✅ Quantize — REAL
`POST /api/quantize` (precision `int8`) is a genuine, tested FP32→INT8
symmetric linear quantization pass implemented directly in the Worker
(`src/quantize.ts`). It does **not** depend on `env.AI`. It is covered by:

- **Unit tests** (`src/quantize.test.ts`, 10 tests) asserting hand-computed
  expected values: input `[1,-1,0.5,-0.5,0.25,-0.25,0,0.125]` →
  `[127,-127,64,-63,32,-32,0,16]`, scale `1/127`, payload 8 B vs 32 B (75%
  reduction), stored blob 16 B vs 32 B (50% reduction), plus dequantization
  round-trip and blob encode/decode validation.
- **Integration tests** (`src/worker.test.ts`, 9 tests) running the real Worker
  in the `workerd` runtime via `@cloudflare/vitest-pool-workers` with in-memory
  KV/R2 stubs, asserting the live HTTP response carries those exact numbers and
  the downloaded blob decodes to the expected INT8 array.

`int4` quantization is **not** implemented and is honestly rejected with `501`
rather than faked.

### 🔮 Compile — still pending a real backend
`POST /api/compile` still calls `env.AI.run("@cf/onnx", ...)`, a model ID that
does **not** exist in Cloudflare's real catalog. The surrounding scaffolding
(validation, KV job tracking, R2 storage, hardware-target matrix) is real and
functional, but the actual compilation step is 🔮 and will fail at runtime with
"model not found" until either (a) a real model-transformation backend ships in
Workers AI, or (b) a custom inference/compilation endpoint is wired in. The
placeholder is deliberately retained rather than faked.

## 🚀 Deployment

After updating `wrangler.toml` with the required bindings, run:

```bash
npx wrangler deploy
```

## 🧪 Testing

```bash
npm run typecheck   # tsc, strict
npm test            # vitest: 19 tests (10 unit + 9 integration)
```

Tests run the real Worker inside the `workerd` runtime via
`@cloudflare/vitest-pool-workers` (`vitest.config.mts`) with in-memory KV/R2
stubs — no Cloudflare account or network is required. The test-only
`wrangler.test.toml` intentionally omits the `[ai]` binding so no remote Workers
AI connection is opened during tests.

## License

This project is part of the [Cocapn fleet](https://github.com/Lucineer/the-fleet).
