import { describe, it, expect } from "vitest";
import {
  quantizeFloat32ToInt8,
  dequantizeInt8ToFloat32,
  encodeQuantizedBlob,
  decodeQuantizedBlob,
  sizeReductionFraction,
} from "./quantize";

/**
 * Hand-computed reference. Every expected value below was derived by hand from
 * the symmetric INT8 algorithm (scale = absmax / 127, q = round(x / scale)),
 * using JavaScript Math.round semantics (halves round toward +Infinity).
 *
 *   input          x * 127      round(x*127)
 *   1.0            127.0        127
 *  -1.0           -127.0       -127
 *   0.5             63.5         64
 *  -0.5            -63.5        -63
 *   0.25            31.75        32
 *  -0.25           -31.75       -32
 *   0.0              0.0          0
 *   0.125           15.875       16
 *
 * absmax = 1.0  =>  scale = 1 / 127 = 0.007874015748031496
 * originalBytes = 8 * 4 = 32
 * quantizedPayloadBytes = 8 * 1 = 8   => payload reduction = 1 - 8/32 = 0.75
 * blobBytes = 8 (header) + 8 (payload) = 16  => blob reduction = 1 - 16/32 = 0.5
 */
const INPUT = new Float32Array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.0, 0.125]);
const EXPECTED_INT8 = [127, -127, 64, -63, 32, -32, 0, 16];
const EXPECTED_SCALE = 1 / 127; // 0.007874015748031496

describe("quantizeFloat32ToInt8", () => {
  it("produces the hand-computed INT8 values exactly", () => {
    const result = quantizeFloat32ToInt8(INPUT);
    expect(Array.from(result.quantized)).toEqual(EXPECTED_INT8);
  });

  it("computes the symmetric scale as absmax / 127", () => {
    const result = quantizeFloat32ToInt8(INPUT);
    expect(result.scale).toBeCloseTo(EXPECTED_SCALE, 10);
    expect(result.zeroPoint).toBe(0);
  });

  it("reduces payload size by exactly 4x (75%)", () => {
    const result = quantizeFloat32ToInt8(INPUT);
    expect(result.originalBytes).toBe(32);
    expect(result.quantizedPayloadBytes).toBe(8);
    expect(
      sizeReductionFraction(result.originalBytes, result.quantizedPayloadBytes),
    ).toBe(0.75);
  });

  it("handles an all-zero tensor without NaN (scale guard)", () => {
    const zeros = new Float32Array([0, 0, 0, 0]);
    const result = quantizeFloat32ToInt8(zeros);
    expect(result.scale).toBe(1);
    expect(Array.from(result.quantized)).toEqual([0, 0, 0, 0]);
    expect(Number.isNaN(result.scale)).toBe(false);
  });
});

describe("dequantizeInt8ToFloat32", () => {
  it("recovers the hand-computed dequantized values (q * scale)", () => {
    const { quantized, scale } = quantizeFloat32ToInt8(INPUT);
    const deq = dequantizeInt8ToFloat32(quantized, scale);
    // q * (1/127), computed by hand:
    //  127/127 = 1.0
    // -127/127 = -1.0
    //   64/127 = 0.5039370078740157
    //  -63/127 = -0.49606299212598426
    //   32/127 = 0.25196850393700787
    //  -32/127 = -0.25196850393700787
    //    0/127 = 0
    //   16/127 = 0.12598425196850394
    const expected = [
      1.0,
      -1.0,
      64 / 127,
      -63 / 127,
      32 / 127,
      -32 / 127,
      0,
      16 / 127,
    ];
    expect(deq.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(deq[i]).toBeCloseTo(expected[i], 6);
    }
  });

  it("keeps max quantization error under one scale step (~1/127)", () => {
    const { quantized, scale } = quantizeFloat32ToInt8(INPUT);
    const deq = dequantizeInt8ToFloat32(quantized, scale);
    let maxErr = 0;
    for (let i = 0; i < INPUT.length; i++) {
      const err = Math.abs(deq[i] - INPUT[i]);
      if (err > maxErr) maxErr = err;
    }
    // Worst-case rounding error is half a scale step = 1/(2*127) ~ 0.00394.
    expect(maxErr).toBeLessThan(scale);
  });
});

describe("quantized blob encode/decode", () => {
  it("encodes to a 16-byte blob (8 header + 8 payload) => 50% reduction", () => {
    const result = quantizeFloat32ToInt8(INPUT);
    const blob = encodeQuantizedBlob(result);
    expect(blob.byteLength).toBe(16);
    expect(sizeReductionFraction(result.originalBytes, blob.byteLength)).toBe(0.5);
  });

  it("round-trips: decode(blob) recovers the exact INT8 payload and scale", () => {
    const result = quantizeFloat32ToInt8(INPUT);
    const blob = encodeQuantizedBlob(result);
    const decoded = decodeQuantizedBlob(blob);
    expect(decoded.magic).toBe("QNT1");
    expect(Array.from(decoded.quantized)).toEqual(EXPECTED_INT8);
    // scale is stored as float32, so allow small precision loss
    expect(decoded.scale).toBeCloseTo(EXPECTED_SCALE, 5);
  });

  it("rejects a blob with a bad magic bytes", () => {
    const result = quantizeFloat32ToInt8(INPUT);
    const blob = encodeQuantizedBlob(result);
    blob[0] = 0x00; // corrupt magic
    expect(() => decodeQuantizedBlob(blob)).toThrow(/bad magic/);
  });

  it("rejects a blob that is too short", () => {
    expect(() => decodeQuantizedBlob(new Uint8Array(4))).toThrow(/too short/);
  });
});
