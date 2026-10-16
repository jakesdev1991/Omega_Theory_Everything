// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

export const LEGACY_OPERATOR_KEY_STORAGE = "omega.store.operatorKey.v1";

export interface OperatorKeyStorage {
  getItem(key: string): string | null;
  removeItem(key: string): void;
}

export interface SessionOperatorKeyResult {
  secret: Uint8Array;
  hadLegacyValue: boolean;
  restoredLegacyValue: boolean;
  removedLegacyValue: boolean;
}

/**
 * Recover and immediately remove the old plaintext localStorage key, if present.
 * New keys are generated in memory only; this helper never persists a secret.
 */
export function loadSessionOperatorKey(
  storage: OperatorKeyStorage | null,
  decodeSecret: (value: string) => Uint8Array | null,
  generateSecret: () => Uint8Array,
): SessionOperatorKeyResult {
  let legacyValue: string | null = null;
  let secret: Uint8Array | null = null;
  let removedLegacyValue = true;
  let hadLegacyValue = false;

  if (storage) {
    try {
      legacyValue = storage.getItem(LEGACY_OPERATOR_KEY_STORAGE);
      hadLegacyValue = legacyValue !== null;
      if (legacyValue) secret = decodeSecret(legacyValue);
      if (hadLegacyValue) {
        storage.removeItem(LEGACY_OPERATOR_KEY_STORAGE);
        removedLegacyValue = storage.getItem(LEGACY_OPERATOR_KEY_STORAGE) === null;
      }
    } catch {
      removedLegacyValue = false;
    }
  } else {
    removedLegacyValue = false;
  }

  return {
    secret: secret ?? generateSecret(),
    hadLegacyValue,
    restoredLegacyValue: secret !== null,
    removedLegacyValue,
  };
}
