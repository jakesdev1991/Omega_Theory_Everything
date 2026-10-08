// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

/**
 * Tiny ICO encoder (classic 32-bit DIB entries, no external dependencies).
 *
 * Why this exists at all: the Windows half of the desktop wrapper does not build
 * without a real `.ico`. `tauri-build` reads one into the application resource
 * during `cargo build` — it searches `bundle > icon` in `tauri.conf.json` for an
 * entry ending in `.ico`, falls back to `src-tauri/icons/icon.ico`, and hard-errors
 * when neither exists — and the MSI bundler uses it for the installer and shortcut
 * icons. The wallet's icons are generated, not committed, so the generated icon
 * set now includes an `.ico` as well.
 *
 * Entries are written in the BITMAPINFOHEADER + XOR bitmap + AND mask form rather
 * than the PNG-compressed form some tools emit: PNG entries are understood by the
 * Windows Vista+ icon *loader*, not necessarily by the resource compiler that
 * embeds the icon in the binary. The AND mask is zeroed because transparency is
 * carried by the 32-bit alpha channel, which is what both use.
 */

const ICONDIR_BYTES = 6;
const ICONDIRENTRY_BYTES = 16;
const BITMAPINFOHEADER_BYTES = 40;

/**
 * One icon image in DIB form: header, bottom-up BGRA rows, then the 1-bit AND
 * mask (rows padded to 4 bytes).
 *
 * @param {number} size
 * @param {Buffer} rgba top-down RGBA, size * size * 4
 * @returns {Buffer}
 */
function bmpEntry(size, rgba) {
  if (rgba.length !== size * size * 4) {
    throw new Error(`RGBA buffer length ${rgba.length} does not match ${size}x${size}`);
  }

  const maskStride = Math.ceil(size / 32) * 4;
  const xorBytes = size * size * 4;
  const maskBytes = maskStride * size;

  const header = Buffer.alloc(BITMAPINFOHEADER_BYTES);
  header.writeUInt32LE(BITMAPINFOHEADER_BYTES, 0);
  header.writeInt32LE(size, 4); // biWidth
  header.writeInt32LE(size * 2, 8); // biHeight: XOR bitmap stacked above AND mask
  header.writeUInt16LE(1, 12); // biPlanes
  header.writeUInt16LE(32, 14); // biBitCount
  header.writeUInt32LE(0, 16); // biCompression: BI_RGB
  header.writeUInt32LE(xorBytes + maskBytes, 20); // biSizeImage

  const xor = Buffer.alloc(xorBytes);
  for (let y = 0; y < size; y += 1) {
    const sourceRow = (size - 1 - y) * size * 4; // DIB rows are bottom-up
    for (let x = 0; x < size; x += 1) {
      const from = sourceRow + x * 4;
      const to = (y * size + x) * 4;
      xor[to] = rgba[from + 2]; // blue
      xor[to + 1] = rgba[from + 1]; // green
      xor[to + 2] = rgba[from]; // red
      xor[to + 3] = rgba[from + 3]; // alpha
    }
  }

  return Buffer.concat([header, xor, Buffer.alloc(maskBytes)]);
}

/**
 * @param {number[]} sizes ascending, each <= 256 (256 is stored as 0 per spec)
 * @param {(size: number) => Buffer} draw returns top-down RGBA for a size
 * @returns {Buffer} ICO bytes
 */
export function encodeIco(sizes, draw) {
  const entries = sizes.map((size) => ({ size, data: bmpEntry(size, draw(size)) }));

  const directory = Buffer.alloc(ICONDIR_BYTES + ICONDIRENTRY_BYTES * entries.length);
  directory.writeUInt16LE(0, 0); // reserved
  directory.writeUInt16LE(1, 2); // type: 1 = icon
  directory.writeUInt16LE(entries.length, 4);

  let offset = directory.length;
  entries.forEach((entry, index) => {
    const at = ICONDIR_BYTES + index * ICONDIRENTRY_BYTES;
    const dimension = entry.size >= 256 ? 0 : entry.size;
    directory[at] = dimension;
    directory[at + 1] = dimension;
    directory[at + 2] = 0; // palette size
    directory[at + 3] = 0; // reserved
    directory.writeUInt16LE(1, at + 4); // planes
    directory.writeUInt16LE(32, at + 6); // bit count
    directory.writeUInt32LE(entry.data.length, at + 8);
    directory.writeUInt32LE(offset, at + 12);
    offset += entry.data.length;
  });

  return Buffer.concat([directory, ...entries.map((entry) => entry.data)]);
}
