// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
import "server-only";

import { deflateRawSync } from "node:zlib";

/**
 * Minimal, dependency-free ZIP writer used to package the Omega wallet GUI as a
 * downloadable offline bundle.
 *
 * Deliberate constraints:
 * - Only "stored" (0) and "deflate" (8) entries, no ZIP64, no encryption, no
 *   directories entries. Bundles are small (a few hundred KB) and always fit.
 * - Timestamps are pinned to a caller-supplied DOS date/time so the produced
 *   bytes are deterministic. That matters because the download page publishes a
 *   SHA-256 for the bundle: the same inputs must always produce the same hash.
 */

export interface ZipEntry {
  /** Path inside the archive, forward slashes, no leading slash. */
  path: string;
  data: Uint8Array;
  /** When false the entry is stored uncompressed. Defaults to true. */
  deflate?: boolean;
}

export interface ZipOptions {
  /** DOS time word. Defaults to 0 (00:00:00) for deterministic output. */
  dosTime?: number;
  /** DOS date word. Defaults to 2026-01-01 for deterministic output. */
  dosDate?: number;
}

const LOCAL_FILE_HEADER_SIGNATURE = 0x04034b50;
const CENTRAL_DIRECTORY_SIGNATURE = 0x02014b50;
const END_OF_CENTRAL_DIRECTORY_SIGNATURE = 0x06054b50;
const VERSION_NEEDED = 20;
const VERSION_MADE_BY = 20;
const DEFAULT_DOS_DATE = ((2026 - 1980) << 9) | (1 << 5) | 1;

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

export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let index = 0; index < data.length; index += 1) {
    crc = CRC_TABLE[(crc ^ data[index]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function assertValidPath(path: string) {
  if (!path || path.startsWith("/") || path.includes("\\") || path.split("/").some((part) => part === "..")) {
    throw new Error(`Refusing to add unsafe archive path: ${path}`);
  }
}

export function buildZip(entries: ZipEntry[], options: ZipOptions = {}): Uint8Array {
  const dosTime = options.dosTime ?? 0;
  const dosDate = options.dosDate ?? DEFAULT_DOS_DATE;

  const localChunks: Uint8Array[] = [];
  const centralChunks: Uint8Array[] = [];
  let offset = 0;

  for (const entry of entries) {
    assertValidPath(entry.path);

    const nameBytes = new TextEncoder().encode(entry.path);
    const raw = entry.data;
    const useDeflate = entry.deflate !== false;
    const compressed = useDeflate ? deflateRawSync(Buffer.from(raw)) : Buffer.from(raw);
    const method = useDeflate && compressed.length < raw.length ? 8 : 0;
    const payload = method === 8 ? new Uint8Array(compressed) : raw;
    const checksum = crc32(raw);

    const localHeader = Buffer.alloc(30);
    localHeader.writeUInt32LE(LOCAL_FILE_HEADER_SIGNATURE, 0);
    localHeader.writeUInt16LE(VERSION_NEEDED, 4);
    localHeader.writeUInt16LE(0, 6); // flags
    localHeader.writeUInt16LE(method, 8);
    localHeader.writeUInt16LE(dosTime, 10);
    localHeader.writeUInt16LE(dosDate, 12);
    localHeader.writeUInt32LE(checksum, 14);
    localHeader.writeUInt32LE(payload.length, 18);
    localHeader.writeUInt32LE(raw.length, 22);
    localHeader.writeUInt16LE(nameBytes.length, 26);
    localHeader.writeUInt16LE(0, 28); // extra length

    localChunks.push(new Uint8Array(localHeader), nameBytes, payload);

    const centralHeader = Buffer.alloc(46);
    centralHeader.writeUInt32LE(CENTRAL_DIRECTORY_SIGNATURE, 0);
    centralHeader.writeUInt16LE(VERSION_MADE_BY, 4);
    centralHeader.writeUInt16LE(VERSION_NEEDED, 6);
    centralHeader.writeUInt16LE(0, 8); // flags
    centralHeader.writeUInt16LE(method, 10);
    centralHeader.writeUInt16LE(dosTime, 12);
    centralHeader.writeUInt16LE(dosDate, 14);
    centralHeader.writeUInt32LE(checksum, 16);
    centralHeader.writeUInt32LE(payload.length, 20);
    centralHeader.writeUInt32LE(raw.length, 24);
    centralHeader.writeUInt16LE(nameBytes.length, 28);
    centralHeader.writeUInt16LE(0, 30); // extra length
    centralHeader.writeUInt16LE(0, 32); // comment length
    centralHeader.writeUInt16LE(0, 34); // disk number start
    centralHeader.writeUInt16LE(0, 36); // internal attributes
    centralHeader.writeUInt32LE(0o644 << 16, 38); // external attributes
    centralHeader.writeUInt32LE(offset, 42);

    centralChunks.push(new Uint8Array(centralHeader), nameBytes);

    offset += localHeader.length + nameBytes.length + payload.length;
  }

  const centralDirectory = Buffer.concat(centralChunks.map((chunk) => Buffer.from(chunk)));

  const endRecord = Buffer.alloc(22);
  endRecord.writeUInt32LE(END_OF_CENTRAL_DIRECTORY_SIGNATURE, 0);
  endRecord.writeUInt16LE(0, 4); // disk number
  endRecord.writeUInt16LE(0, 6); // disk with central directory
  endRecord.writeUInt16LE(entries.length, 8);
  endRecord.writeUInt16LE(entries.length, 10);
  endRecord.writeUInt32LE(centralDirectory.length, 12);
  endRecord.writeUInt32LE(offset, 16);
  endRecord.writeUInt16LE(0, 20); // comment length

  return new Uint8Array(
    Buffer.concat([...localChunks.map((chunk) => Buffer.from(chunk)), centralDirectory, endRecord]),
  );
}
