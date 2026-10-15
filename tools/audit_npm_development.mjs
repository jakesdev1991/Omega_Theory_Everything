#!/usr/bin/env node
// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: LicenseRef-Omega-ReadOnly
/**
 * Ratchet explicitly documented npm development-tooling audit exceptions.
 *
 * Production trees are checked separately by audit_npm_production.mjs. This
 * command intentionally includes development dependencies for the small set of
 * reviewed exceptions in docs/security/npm-development-audit-exceptions.json.
 * It fails on a new advisory, a changed dependency path, an expired review, or
 * a stale exception after the upstream issue disappears. An exception is a
 * time-bounded review record, never a blanket npm-audit suppression.
 *
 * Run from the repository root:
 *   node tools/audit_npm_development.mjs
 */

import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";

const ROOT = resolve(new URL("..", import.meta.url).pathname);
const DEFAULT_EXCEPTIONS_PATH = resolve(ROOT, "docs/security/npm-development-audit-exceptions.json");
const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";

function fail(message) {
  console.error(`Development dependency audit failed: ${message}`);
  process.exitCode = 1;
}

function exceptionsPathFromArgs(args) {
  if (!args.length) {
    return DEFAULT_EXCEPTIONS_PATH;
  }
  if (args.length === 2 && args[0] === "--exceptions") {
    return resolve(ROOT, args[1]);
  }
  throw new Error("usage: node tools/audit_npm_development.mjs [--exceptions <path>]");
}

function readExceptions(exceptionsPath) {
  let document;
  try {
    document = JSON.parse(readFileSync(exceptionsPath, "utf8"));
  } catch (error) {
    throw new Error(`cannot read exception register: ${error.message}`);
  }

  if (document.schema_version !== 1 || !Array.isArray(document.exceptions) || !document.exceptions.length) {
    throw new Error("exception register must contain schema_version 1 and a non-empty exceptions list");
  }
  return document.exceptions;
}

function advisoryNames(via) {
  if (!Array.isArray(via)) {
    return [];
  }
  return via.map((item) => (typeof item === "string" ? item : item?.name)).filter(Boolean).sort();
}

function sameStrings(actual, expected) {
  return JSON.stringify([...actual].sort()) === JSON.stringify([...expected].sort());
}

function ensureExceptionShape(exception) {
  const requiredStrings = [
    "id",
    "workspace",
    "advisory",
    "advisory_package",
    "severity",
    "root_dev_dependency",
    "review_by",
    "owner_role",
    "reason",
  ];
  for (const field of requiredStrings) {
    if (typeof exception[field] !== "string" || !exception[field].trim()) {
      throw new Error(`${exception.id ?? "unknown exception"}: ${field} must be a non-empty string`);
    }
  }
  if (!Array.isArray(exception.affected_packages) || !exception.affected_packages.length) {
    throw new Error(`${exception.id}: affected_packages must be a non-empty list`);
  }
  if (typeof exception.expected_via !== "object" || exception.expected_via === null) {
    throw new Error(`${exception.id}: expected_via must be an object`);
  }
  if (!sameStrings(Object.keys(exception.expected_via), exception.affected_packages)) {
    throw new Error(`${exception.id}: expected_via must cover exactly affected_packages`);
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(exception.review_by)) {
    throw new Error(`${exception.id}: review_by must use YYYY-MM-DD`);
  }
}

function auditException(exception) {
  ensureExceptionShape(exception);
  const today = new Date().toISOString().slice(0, 10);
  if (exception.review_by < today) {
    throw new Error(`${exception.id}: review expired on ${exception.review_by}`);
  }

  const cwd = resolve(ROOT, exception.workspace);
  const lockfile = resolve(cwd, "package-lock.json");
  const packageJson = resolve(cwd, "package.json");
  if (!existsSync(lockfile) || !existsSync(packageJson)) {
    throw new Error(`${exception.id}: ${exception.workspace} must contain package.json and package-lock.json`);
  }

  const manifest = JSON.parse(readFileSync(packageJson, "utf8"));
  if (!manifest.devDependencies?.[exception.root_dev_dependency]) {
    throw new Error(
      `${exception.id}: ${exception.root_dev_dependency} is no longer a root development dependency`,
    );
  }

  console.log(`\n=== ${exception.workspace}: bounded development dependency audit ===`);
  const result = spawnSync(npmCommand, ["audit", "--package-lock-only", "--json"], {
    cwd,
    encoding: "utf8",
  });
  if (result.error) {
    throw new Error(`${exception.id}: unable to run npm audit: ${result.error.message}`);
  }

  let audit;
  try {
    audit = JSON.parse(result.stdout);
  } catch (error) {
    throw new Error(`${exception.id}: npm audit did not return JSON: ${error.message}`);
  }
  if (audit.error) {
    throw new Error(`${exception.id}: npm audit returned ${audit.error.summary ?? "an error"}`);
  }
  if (result.status !== 0 && result.status !== 1) {
    throw new Error(`${exception.id}: npm audit exited ${result.status}`);
  }

  const vulnerabilities = audit.vulnerabilities ?? {};
  const actualPackages = Object.keys(vulnerabilities).sort();
  if (!sameStrings(actualPackages, exception.affected_packages)) {
    throw new Error(
      `${exception.id}: unexpected vulnerability package set: ${JSON.stringify(actualPackages)}`,
    );
  }

  for (const packageName of exception.affected_packages) {
    const vulnerability = vulnerabilities[packageName];
    if (vulnerability?.severity !== exception.severity) {
      throw new Error(
        `${exception.id}: ${packageName} severity changed to ${vulnerability?.severity ?? "missing"}`,
      );
    }
    const expectedVia = exception.expected_via[packageName];
    if (!Array.isArray(expectedVia) || !sameStrings(advisoryNames(vulnerability.via), expectedVia)) {
      throw new Error(
        `${exception.id}: ${packageName} dependency path changed to ${JSON.stringify(advisoryNames(vulnerability.via))}`,
      );
    }
  }

  const advisory = vulnerabilities[exception.advisory_package]?.via?.find(
    (item) =>
      typeof item === "object" &&
      typeof item?.url === "string" &&
      item.url.includes(exception.advisory),
  );
  if (!advisory) {
    throw new Error(
      `${exception.id}: expected advisory ${exception.advisory} is absent from ${exception.advisory_package}`,
    );
  }
  console.log(
    `Reviewed exception ${exception.id} remains bounded to development tooling; review by ${exception.review_by}.`,
  );
}

try {
  const exceptions = readExceptions(exceptionsPathFromArgs(process.argv.slice(2)));
  const ids = exceptions.map((exception) => exception.id);
  if (new Set(ids).size !== ids.length) {
    throw new Error("exception register contains duplicate ids");
  }
  for (const exception of exceptions) {
    auditException(exception);
  }
  console.log("\nAll documented npm development dependency exceptions match the reviewed state.");
} catch (error) {
  fail(error.message);
}
