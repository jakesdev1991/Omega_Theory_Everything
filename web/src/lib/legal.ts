// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
/**
 * Loads and renders the store's legal documents (web/public/legal/*.md).
 *
 * The markdown files are the canonical texts (also downloadable raw from
 * /legal/*.md). Rendering is a deliberately small, escape-first subset:
 * headings, paragraphs, blockquotes, ordered/unordered lists, pipe tables,
 * **bold**, `code` and [links](https://…). Everything is HTML-escaped before
 * inline formatting is applied, and only http(s)/relative links are emitted.
 */

import fs from "node:fs";
import path from "node:path";

export interface LegalDocument {
  slug: string;
  title: string;
  html: string;
  raw: string;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function inline(text: string): string {
  let out = escapeHtml(text);
  out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
  out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label: string, href: string) => {
    const safe = /^(https?:\/\/|\/|#)/.test(href) ? href : "#";
    const external = safe.startsWith("http");
    return `<a href="${safe}"${external ? ' rel="noopener noreferrer" target="_blank"' : ""}>${label}</a>`;
  });
  return out;
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");
}

export function renderLegalMarkdown(markdown: string, idPrefix = ""): string {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html: string[] = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      const level = heading[1].length;
      html.push(`<h${level} id="${idPrefix}${slugify(heading[2])}">${inline(heading[2])}</h${level}>`);
      index += 1;
      continue;
    }

    if (line.startsWith(">")) {
      const quote: string[] = [];
      while (index < lines.length && lines[index].startsWith(">")) {
        quote.push(lines[index].replace(/^>\s?/, ""));
        index += 1;
      }
      html.push(`<blockquote><p>${inline(quote.join(" "))}</p></blockquote>`);
      continue;
    }

    if (line.startsWith("|")) {
      const rows: string[][] = [];
      while (index < lines.length && lines[index].startsWith("|")) {
        const cells = lines[index].trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
        if (!cells.every((cell) => /^:?-{2,}:?$/.test(cell))) rows.push(cells);
        index += 1;
      }
      const [head, ...body] = rows;
      html.push(
        `<table><thead><tr>${head.map((cell) => `<th>${inline(cell)}</th>`).join("")}</tr></thead><tbody>${body
          .map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`)
          .join("")}</tbody></table>`,
      );
      continue;
    }

    const listMatch = line.match(/^(\s*)(\d+\.|-)\s+/);
    if (listMatch) {
      const ordered = listMatch[2] !== "-";
      const items: string[] = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s*(\d+\.|-)\s+(.*)$/);
        if (!item || (item[1] === "-") === ordered) break;
        items.push(item[2]);
        index += 1;
      }
      const tag = ordered ? "ol" : "ul";
      html.push(`<${tag}>${items.map((item) => `<li>${inline(item)}</li>`).join("")}</${tag}>`);
      continue;
    }

    const paragraph: string[] = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^(#{1,4}\s|>|\||\s*(\d+\.|-)\s)/.test(lines[index])
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    html.push(`<p>${inline(paragraph.join(" "))}</p>`);
  }

  return html.join("\n");
}

export const LEGAL_DOCUMENTS = [
  { slug: "store-terms", file: "store-terms.md" },
  { slug: "publisher-agreement", file: "publisher-agreement.md" },
] as const;

export function loadLegalDocument(file: string): LegalDocument {
  const raw = fs.readFileSync(path.join(process.cwd(), "public", "legal", file), "utf-8");
  const title = raw.match(/^#\s+(.*)$/m)?.[1] ?? file;
  const slug = file.replace(/\.md$/, "");
  return { slug, title, html: renderLegalMarkdown(raw, `${slug}-`), raw };
}
