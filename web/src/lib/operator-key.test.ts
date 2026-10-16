// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import assert from "node:assert/strict";
import test from "node:test";

import { LEGACY_OPERATOR_KEY_STORAGE, loadSessionOperatorKey } from "./operator-key";

function memoryStorage(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    values,
    getItem(key: string) {
      return values.get(key) ?? null;
    },
    removeItem(key: string) {
      values.delete(key);
    },
  };
}

const validSecret = new Uint8Array(32).fill(7);
const decodeSecret = (value: string) => (value === "valid-nsec" ? validSecret : null);

test("recovers a legacy plaintext key in memory and removes its persistent copy", () => {
  const storage = memoryStorage({ [LEGACY_OPERATOR_KEY_STORAGE]: "valid-nsec" });
  let generated = false;
  const result = loadSessionOperatorKey(storage, decodeSecret, () => {
    generated = true;
    return new Uint8Array(32).fill(9);
  });

  assert.equal(result.secret, validSecret);
  assert.equal(result.hadLegacyValue, true);
  assert.equal(result.restoredLegacyValue, true);
  assert.equal(result.removedLegacyValue, true);
  assert.equal(storage.getItem(LEGACY_OPERATOR_KEY_STORAGE), null);
  assert.equal(generated, false);
});

test("drops an invalid legacy value and generates a fresh session-only key", () => {
  const storage = memoryStorage({ [LEGACY_OPERATOR_KEY_STORAGE]: "not-a-secret" });
  const generatedSecret = new Uint8Array(32).fill(3);
  const result = loadSessionOperatorKey(storage, decodeSecret, () => generatedSecret);

  assert.equal(result.secret, generatedSecret);
  assert.equal(result.hadLegacyValue, true);
  assert.equal(result.restoredLegacyValue, false);
  assert.equal(result.removedLegacyValue, true);
  assert.equal(storage.getItem(LEGACY_OPERATOR_KEY_STORAGE), null);
  assert.deepEqual([...storage.values.keys()], []);
});

test("reports a legacy value that browser storage refuses to remove", () => {
  const storage = {
    getItem: () => "valid-nsec",
    removeItem: () => undefined,
  };
  const result = loadSessionOperatorKey(storage, decodeSecret, () => new Uint8Array(32).fill(8));

  assert.equal(result.secret, validSecret);
  assert.equal(result.hadLegacyValue, true);
  assert.equal(result.restoredLegacyValue, true);
  assert.equal(result.removedLegacyValue, false);
});

test("does not persist a generated key when storage is unavailable", () => {
  const generatedSecret = new Uint8Array(32).fill(5);
  const result = loadSessionOperatorKey(null, decodeSecret, () => generatedSecret);

  assert.equal(result.secret, generatedSecret);
  assert.equal(result.hadLegacyValue, false);
  assert.equal(result.restoredLegacyValue, false);
  assert.equal(result.removedLegacyValue, false);
});
