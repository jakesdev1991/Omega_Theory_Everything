// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import { deflateSync } from "node:zlib";

/**
 * Tiny PNG encoder (8-bit RGBA, single IDAT) used to generate the wallet's
 * installable-app icons without pulling an image library into the web package.
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let index = 0; index < 256; index += 1) {
    let value = index;
    for (let bit = 0; bit < 8; bit += 1) {
      value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    }
    table[index] = value >>> 0;
  }
  return table;
})();

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const typeBuffer = Buffer.from(type, "ascii");
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])), 0);
  return Buffer.concat([length, typeBuffer, data, crc]);
}

/**
 * @param {number} width
 * @param {number} height
 * @param {Uint8Array|Buffer} rgba length must be width * height * 4
 * @returns {Buffer}
 */
export function encodePng(width, height, rgba) {
  if (rgba.length !== width * height * 4) {
    throw new Error(`RGBA buffer length ${rgba.length} does not match ${width}x${height}`);
  }

  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type: RGBA
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const stride = 1 + width * 4;
  const raw = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y += 1) {
    raw[y * stride] = 0; // filter type: none
    rgba.copy(raw, y * stride + 1, y * width * 4, (y + 1) * width * 4);
  }

  return Buffer.concat([
    signature,
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/**
 * Draws the Omega wallet app icon: a dark rounded tile with an amber ring and an
 * Omega glyph cut out of it. Pure per-pixel math, no dependencies.
 *
 * @param {number} size
 * @returns {Buffer} PNG bytes
 */
export function renderOmegaIcon(size) {
  const pixels = Buffer.alloc(size * size * 4);
  const center = size / 2;
  const radius = size * 0.44;
  const cornerRadius = size * 0.22;

  const background = [7, 8, 12];
  const amber = [251, 191, 36];
  const amberSoft = [212, 165, 116];

  const insideRoundedTile = (x, y) => {
    const dx = Math.max(Math.abs(x - center) - (center - cornerRadius), 0);
    const dy = Math.max(Math.abs(y - center) - (center - cornerRadius), 0);
    return Math.hypot(dx, dy) <= cornerRadius;
  };

  // Omega glyph: a thick ring open at the bottom, sitting on two short feet.
  const glyphCenterY = center * 0.92;
  const ringOuter = radius * 0.8;
  const ringInner = radius * 0.52;
  const footTop = glyphCenterY + ringInner * 0.72;
  const footBottom = footTop + size * 0.075;
  const footHalfWidth = size * 0.11;
  const footOffsetX = ringOuter * 0.62;

  const omegaCoverage = (x, y) => {
    const dx = x - center;
    const dy = y - glyphCenterY;
    const distance = Math.hypot(dx, dy);
    const angle = Math.atan2(dy, dx);
    const inRing = distance >= ringInner && distance <= ringOuter;
    const outsideBottomGap = !(angle >= Math.PI * 0.26 && angle <= Math.PI * 0.74);
    const inLeftFoot =
      y >= footTop && y <= footBottom && x >= center - footOffsetX - footHalfWidth && x <= center - footOffsetX + footHalfWidth;
    const inRightFoot =
      y >= footTop && y <= footBottom && x >= center + footOffsetX - footHalfWidth && x <= center + footOffsetX + footHalfWidth;
    return (inRing && outsideBottomGap) || inLeftFoot || inRightFoot;
  };

  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const index = (y * size + x) * 4;
      if (!insideRoundedTile(x + 0.5, y + 0.5)) {
        pixels[index + 3] = 0;
        continue;
      }

      let [r, g, b] = background;
      const distanceFromCenter = Math.hypot(x + 0.5 - center, y + 0.5 - center) / radius;
      const glow = Math.max(0, 1 - distanceFromCenter) * 0.18;
      r += (amberSoft[0] - r) * glow;
      g += (amberSoft[1] - g) * glow;
      b += (amberSoft[2] - b) * glow;

      if (omegaCoverage(x + 0.5, y + 0.5)) {
        const mix = 0.85 + 0.15 * (1 - Math.abs(y - center) / center);
        r = amber[0] * mix;
        g = amber[1] * mix;
        b = amber[2] * mix;
      }

      pixels[index] = Math.round(r);
      pixels[index + 1] = Math.round(g);
      pixels[index + 2] = Math.round(b);
      pixels[index + 3] = 255;
    }
  }

  return encodePng(size, size, pixels);
}
