// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-Product-Proprietary
/**
 * Minimal bech32 (BIP-173) codec for npub/nsec handling on the mobile node.
 * Deliberately tiny: encode/decode only, no checksum-less variants.
 */

const CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
const GENERATOR = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];

function polymod(values) {
  let chk = 1;
  for (const value of values) {
    const top = chk >> 25;
    chk = ((chk & 0x1ffffff) << 5) ^ value;
    for (let i = 0; i < 5; i += 1) {
      if ((top >> i) & 1) chk ^= GENERATOR[i];
    }
  }
  return chk;
}

function hrpExpand(hrp) {
  const out = [];
  for (const char of hrp) out.push(char.charCodeAt(0) >> 5);
  out.push(0);
  for (const char of hrp) out.push(char.charCodeAt(0) & 31);
  return out;
}

function createChecksum(hrp, data) {
  const values = [...hrpExpand(hrp), ...data, 0, 0, 0, 0, 0, 0];
  const mod = polymod(values) ^ 1;
  const out = [];
  for (let i = 0; i < 6; i += 1) out.push((mod >> (5 * (5 - i))) & 31);
  return out;
}

function verifyChecksum(hrp, data) {
  return polymod([...hrpExpand(hrp), ...data]) === 1;
}

function convertBits(data, fromBits, toBits, pad) {
  let acc = 0;
  let bits = 0;
  const out = [];
  const maxv = (1 << toBits) - 1;
  for (const value of data) {
    acc = (acc << fromBits) | value;
    bits += fromBits;
    while (bits >= toBits) {
      bits -= toBits;
      out.push((acc >> bits) & maxv);
    }
  }
  if (pad) {
    if (bits > 0) out.push((acc << (toBits - bits)) & maxv);
  } else if (bits >= fromBits || ((acc << (toBits - bits)) & maxv) !== 0) {
    throw new Error("Invalid bech32 padding");
  }
  return out;
}

export function bech32Encode(hrp, bytes) {
  const data = convertBits([...bytes], 8, 5, true);
  const checksum = createChecksum(hrp, data);
  return `${hrp}1${[...data, ...checksum].map((d) => CHARSET[d]).join("")}`;
}

export function bech32Decode(str) {
  const lowered = str.toLowerCase();
  const pos = lowered.lastIndexOf("1");
  if (pos < 1 || pos + 7 > lowered.length) throw new Error("Invalid bech32 string");
  const hrp = lowered.slice(0, pos);
  const data = [];
  for (const char of lowered.slice(pos + 1)) {
    const index = CHARSET.indexOf(char);
    if (index === -1) throw new Error("Invalid bech32 character");
    data.push(index);
  }
  if (!verifyChecksum(hrp, data)) throw new Error("Invalid bech32 checksum");
  const bytes = convertBits(data.slice(0, -6), 5, 8, false);
  return { hrp, bytes: Uint8Array.from(bytes) };
}

export function npubToHex(npub) {
  const { hrp, bytes } = bech32Decode(npub);
  if (hrp !== "npub" || bytes.length !== 32) throw new Error("Not an npub");
  return Buffer.from(bytes).toString("hex");
}

export function hexToNpub(hex) {
  return bech32Encode("npub", Uint8Array.from(Buffer.from(hex, "hex")));
}

export function nsecToHex(nsec) {
  const { hrp, bytes } = bech32Decode(nsec);
  if (hrp !== "nsec" || bytes.length !== 32) throw new Error("Not an nsec");
  return Buffer.from(bytes).toString("hex");
}
