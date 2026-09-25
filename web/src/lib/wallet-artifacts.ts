// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import "server-only";

import { createHash } from "node:crypto";
import { readFile, readdir } from "node:fs/promises";
import { join, relative, resolve } from "node:path";

import configJson from "../../wallet.config.json";
import { buildZip, type ZipEntry } from "./zip";

export interface WalletFileEntry {
  path: string;
  bytes: number;
  sha256: string;
  generated: boolean;
}

export interface WalletDesktopPlatform {
  os: string;
  arch: string;
  status: "published" | "not_published";
  artifact?: string;
  url?: string;
  sha256?: string;
}

export interface WalletManifest {
  ok: true;
  name: string;
  version: string;
  channel: string;
  license: string;
  notice: string;
  mountPath: string;
  synced: boolean;
  generatedAt: string | null;
  launchUrl: string;
  files: WalletFileEntry[];
  webApp: {
    installable: boolean;
    serviceWorker: boolean;
    manifestUrl: string;
    launchUrl: string;
  };
  bundle: {
    fileName: string;
    bytes: number;
    sha256: string;
    downloadUrl: string;
    contents: string[];
  };
  desktop: {
    status: "published" | "not_published";
    platforms: WalletDesktopPlatform[];
    buildWorkflow: string;
    buildInstructions: string[];
  };
}

interface WalletBuildInfo {
  name: string;
  version: string;
  channel: string;
  mountPath: string;
  generatedAt: string;
  files: Array<{ path: string; bytes: number; sha256: string; generated?: boolean }>;
}

interface DesktopReleaseManifest {
  version?: string;
  artifacts?: Array<{ os: string; arch: string; fileName: string; url: string; sha256: string }>;
}

const GENERATED_FILE_NAMES = new Set([
  "manifest.webmanifest",
  "sw.js",
  "install.js",
  "icon.svg",
  "icon-192.png",
  "icon-512.png",
]);

// The wallet GUI is read from disk at request time on purpose; turbopackIgnore
// keeps the deployment tracer from pulling the whole public/ tree into the bundle.
const webRoot = resolve(/* turbopackIgnore: true */ process.cwd());
export const WALLET_CONFIG = configJson;
export const WALLET_PUBLIC_DIR = resolve(/* turbopackIgnore: true */ webRoot, configJson.publicDir);
export const WALLET_MOUNT_PATH = configJson.mountPath.replace(/\/$/, "");
export const WALLET_LAUNCH_URL = `${WALLET_MOUNT_PATH}/index.html`;
export const WALLET_BUNDLE_FILE_NAME = `omega-wallet-${configJson.version}-${configJson.channel}.zip`;
export const WALLET_DOWNLOAD_URL = "/api/wallet/download";

function sha256(bytes: Uint8Array | Buffer): string {
  return createHash("sha256").update(bytes).digest("hex");
}

async function walkWalletDir(dir: string, base = dir): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const absolute = join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walkWalletDir(absolute, base)));
    } else if (entry.isFile() && entry.name !== "BUILDINFO.json") {
      files.push(relative(base, absolute).split("\\").join("/"));
    }
  }
  return files.sort();
}

export async function readWalletBuildInfo(): Promise<WalletBuildInfo | null> {
  try {
    const raw = await readFile(join(WALLET_PUBLIC_DIR, "BUILDINFO.json"), "utf8");
    return JSON.parse(raw) as WalletBuildInfo;
  } catch {
    return null;
  }
}

async function readWalletFiles(): Promise<Map<string, Buffer>> {
  const files = new Map<string, Buffer>();
  for (const relativePath of await walkWalletDir(WALLET_PUBLIC_DIR)) {
    files.set(relativePath, await readFile(join(WALLET_PUBLIC_DIR, relativePath)));
  }
  return files;
}

const SERVE_SCRIPT = `#!/usr/bin/env node
/* Omega Wallet offline launcher — serves this bundle on a local port.
 * No dependencies: node's http module only. Keys never leave this machine. */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.dirname(fileURLToPath(import.meta.url));
const port = Number(process.env.OMEGA_WALLET_PORT || 8787);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".webmanifest": "application/manifest+json",
  ".json": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

http
  .createServer((req, res) => {
    let urlPath = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (urlPath.startsWith("/omega-wallet")) urlPath = urlPath.slice("/omega-wallet".length);
    if (urlPath === "/" || urlPath === "") urlPath = "/index.html";
    const file = path.join(root, path.normalize(urlPath).replace(/^(\\.\\.)/, ""));
    if (!file.startsWith(root)) {
      res.writeHead(403).end("forbidden");
      return;
    }
    fs.readFile(file, (error, data) => {
      if (error) {
        res.writeHead(404).end("not found");
        return;
      }
      res.writeHead(200, {
        "content-type": types[path.extname(file)] || "application/octet-stream",
        "cache-control": "no-store",
      });
      res.end(data);
    });
  })
  .listen(port, "127.0.0.1", () => {
    console.log("Omega Wallet running at http://127.0.0.1:" + port + "/");
  });
`;

function launcherScripts(version: string): Array<{ path: string; data: string; executable?: boolean }> {
  const banner = `Omega Wallet ${version} — offline launcher`;
  return [
    {
      path: "run.sh",
      executable: true,
      data: `#!/bin/sh
# ${banner} (Linux / macOS)
cd "$(dirname "$0")" || exit 1
if command -v node >/dev/null 2>&1; then
  exec node serve.mjs
fi
echo "node not found; falling back to python3 (service worker install may be limited)"
exec python3 -m http.server "\${OMEGA_WALLET_PORT:-8787}" --bind 127.0.0.1
`,
    },
    {
      path: "run.command",
      executable: true,
      data: `#!/bin/sh
# ${banner} (macOS double-click launcher)
cd "$(dirname "$0")" || exit 1
if command -v node >/dev/null 2>&1; then
  node serve.mjs
else
  python3 -m http.server "\${OMEGA_WALLET_PORT:-8787}" --bind 127.0.0.1
fi
open "http://127.0.0.1:\${OMEGA_WALLET_PORT:-8787}/" 2>/dev/null || true
`,
    },
    {
      path: "run.bat",
      data: `@echo off
rem ${banner} (Windows)
cd /d "%~dp0"
where node >nul 2>nul
if %errorlevel%==0 (
  start "" http://127.0.0.1:8787/
  node serve.mjs
) else (
  start "" http://127.0.0.1:8787/
  python -m http.server 8787 --bind 127.0.0.1
)
`,
    },
    { path: "serve.mjs", data: SERVE_SCRIPT },
  ];
}

async function buildBundleEntries(): Promise<Array<{ path: string; data: Buffer }>> {
  const walletFiles = await readWalletFiles();
  const version = configJson.version as string;

  const entries: Array<{ path: string; data: Buffer }> = [];
  for (const [relativePath, data] of walletFiles) {
    entries.push({ path: relativePath, data });
  }

  for (const launcher of launcherScripts(version)) {
    entries.push({ path: launcher.path, data: Buffer.from(launcher.data, "utf8") });
  }

  const licensePath = resolve(/* turbopackIgnore: true */ webRoot, configJson.licenseFile as string);
  try {
    entries.push({ path: "LICENSE.txt", data: await readFile(licensePath) });
  } catch {
    entries.push({
      path: "LICENSE.txt",
      data: Buffer.from("PolyForm-Noncommercial-1.0.0 — free for noncommercial use; commercial use requires a license from Jacob See. See docs/LICENSING.md in the repository.\n", "utf8"),
    });
  }

  const checksumLines = entries
    .map((entry) => `${sha256(entry.data)}  ${entry.path}`)
    .sort()
    .join("\n");

  entries.push({
    path: "CHECKSUMS.txt",
    data: Buffer.from(
      `# SHA-256 checksums for Omega Wallet ${version} (${configJson.channel})\n# Verify with: sha256sum -c CHECKSUMS.txt  (or certutil / shasum -a 256)\n${checksumLines}\n`,
      "utf8",
    ),
  });

  entries.push({
    path: "README.txt",
    data: Buffer.from(
      [
        `${configJson.name} ${version} (${configJson.channel})`,
        ``,
        configJson.notice as string,
        ``,
        `Run it offline:`,
        `  macOS / Linux : ./run.sh        (or double-click run.command on macOS)`,
        `  Windows       : run.bat`,
        `  Then open     : http://127.0.0.1:8787/`,
        ``,
        `Requires Node.js (any modern version) or Python 3 for the local static server.`,
        `The wallet is fully client-side: keys, keystores, and proofs stay in this`,
        `browser profile's local storage. Nothing is uploaded anywhere.`,
        ``,
        `Install as an app: open the served page in Chrome/Edge and use the`,
        `"Install app" button, or your browser's install-PWA menu entry.`,
        ``,
        `Verify integrity:`,
        `  sha256sum -c CHECKSUMS.txt`,
        ``,
        `License: ${"PolyForm-Noncommercial-1.0.0"} (see LICENSE.txt and docs/LICENSING.md).`,
        `Prototype rails only: $OMEGA on Ethereum Sepolia and TWC on Solana Devnet.`,
        `No value-bearing issuance. Not audited. Not for mainnet use.`,
        ``,
      ].join("\n"),
      "utf8",
    ),
  });

  return entries.sort((a, b) => a.path.localeCompare(b.path));
}

export interface WalletBundle {
  fileName: string;
  bytes: Uint8Array;
  sha256: string;
  contents: string[];
}

export async function buildWalletBundle(): Promise<WalletBundle> {
  const entries = await buildBundleEntries();
  const zipEntries: ZipEntry[] = entries.map((entry) => ({
    path: `omega-wallet-${configJson.version}/${entry.path}`,
    data: new Uint8Array(entry.data),
  }));
  const bytes = buildZip(zipEntries);
  return {
    fileName: WALLET_BUNDLE_FILE_NAME,
    bytes,
    sha256: sha256(Buffer.from(bytes)),
    contents: entries.map((entry) => entry.path),
  };
}

async function readDesktopReleases(): Promise<{
  status: "published" | "not_published";
  platforms: WalletDesktopPlatform[];
  manifest: DesktopReleaseManifest | null;
}> {
  const manifestPath = resolve(/* turbopackIgnore: true */ webRoot, (configJson.desktop as { releaseManifest: string }).releaseManifest);
  try {
    const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as DesktopReleaseManifest;
    const artifacts = manifest.artifacts ?? [];
    return {
      status: artifacts.length > 0 ? "published" : "not_published",
      platforms: artifacts.map((artifact) => ({
        os: artifact.os,
        arch: artifact.arch,
        status: "published" as const,
        artifact: artifact.fileName,
        url: artifact.url,
        sha256: artifact.sha256,
      })),
      manifest,
    };
  } catch {
    return { status: "not_published", platforms: [], manifest: null };
  }
}

export const DESKTOP_BUILD_INSTRUCTIONS = [
  "cd desktop && npm install   # Tauri v2 wrapper around the same wallet GUI",
  "npm run build:macos   # produces a signed-notarized .dmg when Apple credentials exist",
  "npm run build:windows # produces an .msi",
  "npm run build:linux   # produces .AppImage and .deb",
  "Artifacts are uploaded by .github/workflows/wallet-desktop.yml on tag push and listed in desktop/releases.json.",
];

export async function getWalletManifest(): Promise<WalletManifest> {
  const buildInfo = await readWalletBuildInfo();
  const files: WalletFileEntry[] = [];

  if (buildInfo) {
    for (const file of buildInfo.files) {
      files.push({
        path: file.path,
        bytes: file.bytes,
        sha256: file.sha256,
        generated: GENERATED_FILE_NAMES.has(file.path) || file.generated === true,
      });
    }
  } else {
    const walletFiles = await readWalletFiles();
    for (const [relativePath, data] of walletFiles) {
      files.push({
        path: relativePath,
        bytes: data.length,
        sha256: sha256(data),
        generated: GENERATED_FILE_NAMES.has(relativePath),
      });
    }
  }

  const bundle = await buildWalletBundle();
  const desktop = await readDesktopReleases();

  return {
    ok: true,
    name: configJson.name as string,
    version: configJson.version as string,
    channel: configJson.channel as string,
    license: "PolyForm-Noncommercial-1.0.0",
    notice: configJson.notice as string,
    mountPath: WALLET_MOUNT_PATH,
    synced: !!buildInfo,
    generatedAt: buildInfo?.generatedAt ?? null,
    launchUrl: WALLET_LAUNCH_URL,
    files,
    webApp: {
      installable: true,
      serviceWorker: true,
      manifestUrl: `${WALLET_MOUNT_PATH}/manifest.webmanifest`,
      launchUrl: WALLET_LAUNCH_URL,
    },
    bundle: {
      fileName: bundle.fileName,
      bytes: bundle.bytes.length,
      sha256: bundle.sha256,
      downloadUrl: WALLET_DOWNLOAD_URL,
      contents: bundle.contents,
    },
    desktop: {
      status: desktop.status,
      platforms:
        desktop.platforms.length > 0
          ? desktop.platforms
          : [
              { os: "macOS", arch: "universal", status: "not_published" },
              { os: "Windows", arch: "x64", status: "not_published" },
              { os: "Linux", arch: "x64", status: "not_published" },
            ],
      buildWorkflow: (configJson.desktop as { workflow: string }).workflow,
      buildInstructions: DESKTOP_BUILD_INSTRUCTIONS,
    },
  };
}

