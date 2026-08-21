/**
 * Real FP32 -> INT8 symmetric linear quantization.
 *
 * This is a genuine, self-contained quantization pass implemented directly in
 * the Worker. It does NOT depend on any Cloudflare binding (no `env.AI`), so it
 * is fully unit-testable with hand-computed expected values.
 *
 * Algorithm: symmetric per-tensor INT8 quantization.
 *   absmax = max(|x_i|)
 *   scale  = absmax / 127            (so absmax maps to +127)
 *   q_i    = clamp(round(x_i / scale), -128, 127)
 *
 * Rounding follows JavaScript `Math.round` semantics: halves round toward
 * +Infinity (e.g. round(63.5) === 64, round(-63.5) === -63).
 *
 * The quantized payload is stored as a self-describing blob so it can be
 * downloaded and dequantized on its own:
 *   bytes 0..3  : magic "QNT1"
 *   bytes 4..7  : scale as little-endian float32
 *   bytes 8..   : int8 payload
 */

export interface QuantizeResult {
  /** Quantized int8 values, clamped to [-128, 127]. */
  quantized: Int8Array;
  /** Scale factor: dequantized = quantized * scale. */
  scale: number;
  /** Zero point (always 0 for symmetric quantization). */
  zeroPoint: number;
  /** Original input size in bytes (data.byteLength = 4 * length). */
  originalBytes: number;
  /** Quantized payload size in bytes (1 * length). */
  quantizedPayloadBytes: number;
}

const MAGIC = new Uint8Array([0x51, 0x4e, 0x54, 0x31]); // "QNT1"
const MAGIC_STR = "QNT1";
const HEADER_BYTES = 8; // 4 magic + 4 float32 scale
const INT8_MAX = 127;
const INT8_MIN = -128;

export function quantizeFloat32ToInt8(data: Float32Array): QuantizeResult {
  let absmax = 0;
  for (let i = 0; i < data.length; i++) {
    const a = data[i] < 0 ? -data[i] : data[i];
    if (a > absmax) absmax = a;
  }

  // Guard against an all-zero tensor (would divide by zero). With scale 1,
  // every element quantizes to 0, which is the correct INT8 representation.
  const scale = absmax === 0 ? 1 : absmax / INT8_MAX;

  const quantized = new Int8Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const scaled = data[i] / scale;
    const rounded = Math.round(scaled);
    const clamped = rounded > INT8_MAX ? INT8_MAX : rounded < INT8_MIN ? INT8_MIN : rounded;
    quantized[i] = clamped;
  }

  return {
    quantized,
    scale,
    zeroPoint: 0,
    originalBytes: data.byteLength,
    quantizedPayloadBytes: quantized.byteLength,
  };
}

export function dequantizeInt8ToFloat32(
  quantized: Int8Array,
  scale: number,
  zeroPoint = 0,
): Float32Array {
  const out = new Float32Array(quantized.length);
  for (let i = 0; i < quantized.length; i++) {
    out[i] = (quantized[i] - zeroPoint) * scale;
  }
  return out;
}

/** Encode a quantized result as a self-describing, storable blob. */
export function encodeQuantizedBlob(result: QuantizeResult): Uint8Array {
  const blob = new Uint8Array(HEADER_BYTES + result.quantized.byteLength);
  blob.set(MAGIC, 0);
  const dv = new DataView(blob.buffer);
  dv.setFloat32(4, result.scale, true); // little-endian
  blob.set(new Uint8Array(result.quantized.buffer, result.quantized.byteOffset, result.quantized.byteLength), HEADER_BYTES);
  return blob;
}

export interface DecodedBlob {
  quantized: Int8Array;
  scale: number;
  magic: string;
}

/** Decode a self-describing quantization blob. Throws on bad magic/length. */
export function decodeQuantizedBlob(blob: Uint8Array): DecodedBlob {
  if (blob.byteLength < HEADER_BYTES) {
    throw new Error(`Invalid quantized blob: too short (${blob.byteLength} bytes)`);
  }
  const magic = String.fromCharCode(blob[0], blob[1], blob[2], blob[3]);
  if (magic !== MAGIC_STR) {
    throw new Error(`Invalid quantized blob: bad magic "${magic}" (expected "${MAGIC_STR}")`);
  }
  const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const scale = dv.getFloat32(4, true);
  const payload = blob.subarray(HEADER_BYTES);
  const quantized = new Int8Array(payload.byteLength);
  quantized.set(payload);
  return { quantized, scale, magic };
}

/** Size reduction fraction in [0, 1] for a stored blob vs the original input. */
export function sizeReductionFraction(originalBytes: number, blobBytes: number): number {
  if (originalBytes <= 0) return 0;
  return 1 - blobBytes / originalBytes;
}
