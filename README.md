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
Submit a quantization job.

**Request body**
```json
{
  "modelId": "string (required)",
  "precision": "int8 | int4 (required)",
  "calibrationData": "string (optional)"
}
```

**Response (200 OK)**
```json
{
  "modelId": "string",
  "precision": "int8 | int4",
  "sizeReduction": "45.2%",
  "downloadUrl": "/api/download/quantized/<modelId>_<precision>"
}
```

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

| Binding           | Type        | Purpose                           |
|-------------------|-------------|-----------------------------------|
| `COMPILER_CACHE`  | KV Namespace| Track job status                  |
| `MODEL_STORE`     | R2 Bucket   | Store input & output models       |
| `AI`              | Workers AI  | Perform compilation/quantization  |

These are declared as placeholders in `wrangler.toml` — replace the
namespace/bucket IDs with your own before deploying:

```bash
npx wrangler kv namespace create COMPILER_CACHE
npx wrangler r2 bucket create model-store
# Then update wrangler.toml with the generated IDs.
```

## 🔮 Status of the Model‑Transformation Step

The compile (`@cf/onnx`) and quantize (`@cf/quantization`) model IDs called via `env.AI.run()`
**are not confirmed to exist** in Cloudflare Workers AI's real model catalog (which uses
vendor‑prefixed names such as `@cf/meta/llama-3.1-8b-instruct`). The HTTP API scaffolding
(validation, job queuing, KV tracking, R2 storage) is fully functional, but the actual
model‑transformation step is **🔮 pending a real Workers AI model binding**. When a valid
model ID becomes available (or a custom inference endpoint is added), replace the placeholder
names in `compileModel()` and `handleQuantize()`.

## 🚀 Deployment

After updating `wrangler.toml` with the required bindings, run:

```bash
npx wrangler deploy
```

## License

This project is part of the [Cocapn fleet](https://github.com/Lucineer/the-fleet).
