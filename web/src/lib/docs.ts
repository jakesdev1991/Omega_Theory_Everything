// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import fs from "node:fs";
import path from "node:path";

/**
 * C.A.R.E. Economy document loader.
 *
 * Documents live at the repository root (`whitepapers/` and the tri-token
 * blueprint) and are copied into `content/docs/` by `scripts/sync-docs.mjs`
 * (run automatically by `predev` / `prebuild`). This module parses that
 * markdown into structured blocks that render as React components — no
 * raw-HTML interpolation, so the reader is XSS-safe by construction.
 */

export type InlineNode =
  | { kind: "text"; text: string }
  | { kind: "bold"; children: InlineNode[] }
  | { kind: "italic"; children: InlineNode[] }
  | { kind: "code"; text: string }
  | { kind: "link"; href: string; children: InlineNode[]; external?: boolean };

export type DocBlock =
  | { type: "paragraph"; children: InlineNode[] }
  | { type: "heading"; level: 2 | 3 | 4; id: string; text: string; children: InlineNode[] }
  | { type: "list"; ordered: boolean; items: InlineNode[][] }
  | { type: "code"; language: string; text: string }
  | { type: "blockquote"; children: InlineNode[][] }
  | { type: "table"; headers: InlineNode[][]; rows: InlineNode[][][] }
  | { type: "hr" };

export interface DocHeading {
  id: string;
  text: string;
  level: 2 | 3;
}

export interface DocMeta {
  slug: string;
  kind: "Manifesto" | "Whitepaper" | "Blueprint";
  title: string;
  shortTitle: string;
  accent: string;
  summary: string;
  order: number;
}

export interface Doc {
  meta: DocMeta;
  title: string;
  subtitle?: string;
  blocks: DocBlock[];
  toc: DocHeading[];
  wordCount: number;
  readingTimeMinutes: number;
}

/**
 * The documents the website publishes, in the order shown on /whitepapers.
 * Every entry must exist in `scripts/sync-docs.mjs`'s whitelist.
 */
export const DOC_REGISTRY: DocMeta[] = [
  {
    slug: "sovereign_economy_manifesto",
    kind: "Manifesto",
    title: "Manifesto for a Sovereign Agentic Economy",
    shortTitle: "The Manifesto",
    accent: "var(--color-care)",
    summary:
      "The canonical manifesto: a sovereign economy owned by its participants, an agentic framework that stewards rather than rules, and C.A.R.E. — Call About Resuscitating Everyone — as the protected social layer of the whole economy.",
    order: 0,
  },
  {
    slug: "care_economy_whitepaper",
    kind: "Whitepaper",
    title: "The C.A.R.E. Economy — Extensive Whitepaper",
    shortTitle: "Extensive Whitepaper",
    accent: "var(--color-accent)",
    summary:
      "The complete specification of the entire economy: Call About Resuscitating Everyone, five planes, three rails, Proof of Care, Proof of Useful Work, Archangels, privacy, audits, the AMITY boundary, governance, threat model, roadmap, and glossary.",
    order: 1,
  },
  {
    slug: "care_economy_manifesto",
    kind: "Manifesto",
    title: "The C.A.R.E. Economy — Naming and Promises",
    shortTitle: "C.A.R.E. Naming & Promises",
    accent: "var(--color-care)",
    summary:
      "Companion statement: what C.A.R.E. stands for (Call About Resuscitating Everyone), the eight promises, and what the economy refuses to build. The canonical manifesto is the Sovereign Agentic Economy manifesto.",
    order: 2,
  },
  {
    slug: "twc_whitepaper",
    kind: "Whitepaper",
    title: "Token of the World Citizen (TWC) Whitepaper",
    shortTitle: "TWC Whitepaper",
    accent: "var(--color-twc)",
    summary:
      "Useful-work receipts, class-specific verification, appeals, privacy, and the Solana Devnet tTWC boundary. TWC supports the entire crypto world — not one chain or faction.",
    order: 3,
  },
  {
    slug: "care_amity_protocol_whitepaper",
    kind: "Whitepaper",
    title: "C.A.R.E. / AMITY Protocol Whitepaper",
    shortTitle: "C.A.R.E. / AMITY Protocol",
    accent: "var(--color-amity)",
    summary:
      "The protocol detail under the exchange boundary: Proof of Care, CARE Verifiers, Archangels and the 80/20 rule, the thought-virus model, privacy, arbitration, and deployment gates.",
    order: 4,
  },
  {
    slug: "omega_protocol_whitepaper",
    kind: "Whitepaper",
    title: "Omega Protocol Whitepaper",
    shortTitle: "Omega Protocol",
    accent: "var(--color-omega)",
    summary:
      "The operating model for $OMEGA: macro-governance, staking, supply, and the governance plane of the economy.",
    order: 5,
  },
  {
    slug: "lucifer_hermes_omni_bridge_whitepaper",
    kind: "Whitepaper",
    title: "Lucifer–Hermes Omni-Bridge Whitepaper",
    shortTitle: "Omni-Bridge",
    accent: "var(--color-use)",
    summary:
      "Agentic routing, verification, sandboxing, audit, and governance boundaries — how agents are allowed to act inside a human-governed economy.",
    order: 6,
  },
  {
    slug: "tokamak_domain_token_whitepaper",
    kind: "Whitepaper",
    title: "TOKAMAK Domain Token Whitepaper",
    shortTitle: "TOKAMAK (legacy)",
    accent: "var(--color-muted)",
    summary:
      "Legacy plasma-domain computation, telemetry boundaries, and scientific Proof-of-Useful-Work research. TOKAMAK is retired as the Solana token identity.",
    order: 7,
  },
  {
    slug: "tri_token_sovereign_economy_blueprint",
    kind: "Blueprint",
    title: "Tri-Token Sovereign Economy Blueprint",
    shortTitle: "Tri-Token Blueprint",
    accent: "var(--color-unlock)",
    summary:
      "The original systems specification the C.A.R.E. Economy whitepaper builds on: token model, ledger boundaries, PoUW, simulation, clearing, contracts, governance, and delivery plan.",
    order: 8,
  },
];

const DOCS_DIR = path.join(process.cwd(), "content", "docs");

const GITHUB_BASE =
  "https://github.com/jakesdev1991/Omega_Theory_Everything/blob/main";

const KNOWN_SLUGS = new Set(DOC_REGISTRY.map((d) => d.slug));

/** Resolve markdown-relative links to on-site pages or the GitHub source. */
function normalizeLink(href: string): { href: string; external: boolean } {
  if (
    href.startsWith("http://") ||
    href.startsWith("https://") ||
    href.startsWith("#") ||
    href.startsWith("mailto:")
  ) {
    const anchorOnly = href.startsWith("#");
    return { href, external: !anchorOnly };
  }
  if (href.endsWith(".md")) {
    const base = path.posix.basename(href).replace(/\.md$/, "");
    if (KNOWN_SLUGS.has(base)) {
      return { href: `/whitepapers/${base}`, external: false };
    }
    const repoPath = href.replace(/^(\.\.\/)+/, "").replace(/^\.\//, "");
    return { href: `${GITHUB_BASE}/${repoPath}`, external: true };
  }
  return { href, external: false };
}

/* ------------------------------------------------------------------ */
/* Inline markdown parsing                                             */
/* ------------------------------------------------------------------ */

const INLINE_PATTERN =
  /\*\*([^*]+(?:\*(?!\*)[^*]*)*)\*\*|\*([^*\n]+)\*|`([^`\n]+)`|\[([^\]]+)\]\(([^)\s]+)\)/;

function parseInline(text: string): InlineNode[] {
  const nodes: InlineNode[] = [];
  let remaining = text;

  while (remaining.length > 0) {
    const match = INLINE_PATTERN.exec(remaining);
    if (!match || match.index === undefined) {
      nodes.push({ kind: "text", text: remaining });
      break;
    }
    if (match.index > 0) {
      nodes.push({ kind: "text", text: remaining.slice(0, match.index) });
    }
    if (match[1] !== undefined) {
      nodes.push({ kind: "bold", children: parseInline(match[1]) });
    } else if (match[2] !== undefined) {
      nodes.push({ kind: "italic", children: parseInline(match[2]) });
    } else if (match[3] !== undefined) {
      nodes.push({ kind: "code", text: match[3] });
    } else if (match[4] !== undefined && match[5] !== undefined) {
      const { href, external } = normalizeLink(match[5]);
      nodes.push({
        kind: "link",
        href,
        external,
        children: parseInline(match[4]),
      });
    }
    remaining = remaining.slice(match.index + match[0].length);
  }
  return nodes;
}

/* ------------------------------------------------------------------ */
/* Block parsing                                                       */
/* ------------------------------------------------------------------ */

function slugifyHeading(text: string, index: number): string {
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60);
  return `s-${index}-${slug || "section"}`;
}

interface ParseResult {
  title: string;
  subtitle?: string;
  blocks: DocBlock[];
  toc: DocHeading[];
  wordCount: number;
}

export function parseDocument(markdown: string): ParseResult {
  // Strip the leading copyright comment and any other raw HTML comments.
  const cleaned = markdown.replace(/<!--[\s\S]*?-->/g, "").trim();
  const lines = cleaned.split("\n");

  const blocks: DocBlock[] = [];
  const toc: DocHeading[] = [];
  let title = "";
  let subtitle: string | undefined;
  let headingIndex = 0;

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];

    // Fenced code block
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const language = fence[1] || "text";
      const codeLines: string[] = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      i += 1; // skip closing fence
      blocks.push({ type: "code", language, text: codeLines.join("\n") });
      continue;
    }

    // Headings
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      const rawLevel = heading[1].length;
      const text = heading[2].trim();
      const children = parseInline(text);

      if (rawLevel === 1) {
        if (!title) {
          title = text;
        }
        // Subtitle convention: an immediately following standalone italic line.
        const next = lines[i + 1];
        if (next && /^\*[^*]+\*$/.test(next.trim())) {
          subtitle = next.trim().slice(1, -1);
          i += 1;
        }
        i += 1;
        continue;
      }

      const level = (rawLevel >= 5 ? 4 : rawLevel) as 2 | 3 | 4;
      const id = slugifyHeading(text, headingIndex++);
      blocks.push({ type: "heading", level, id, text, children });
      if (level === 2 || level === 3) {
        toc.push({ id, text, level });
      }
      i += 1;
      continue;
    }

    // Horizontal rule
    if (/^\s*(---+|\*\*\*+|___+)\s*$/.test(line)) {
      blocks.push({ type: "hr" });
      i += 1;
      continue;
    }

    // Table (GFM pipe table)
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|\s*$/.test(lines[i + 1])) {
      const splitRow = (row: string): string[] =>
        row
          .trim()
          .replace(/^\|/, "")
          .replace(/\|$/, "")
          .split("|")
          .map((c) => c.trim());
      const headers = splitRow(line).map(parseInline);
      i += 2;
      const rows: InlineNode[][][] = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        rows.push(splitRow(lines[i]).map(parseInline));
        i += 1;
      }
      blocks.push({ type: "table", headers, rows });
      continue;
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const quoteLines: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      blocks.push({
        type: "blockquote",
        children: quoteLines
          .filter((l) => l.trim().length > 0)
          .map((l) => parseInline(l)),
      });
      continue;
    }

    // Unordered list
    if (/^\s*[-*+]\s+/.test(line)) {
      const items: InlineNode[][] = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(parseInline(lines[i].replace(/^\s*[-*+]\s+/, "").trim()));
        i += 1;
        // continuation lines indented under the item
        while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*[-*+]\s+/.test(lines[i])) {
          const last = items[items.length - 1];
          last.push({ kind: "text", text: ` ${lines[i].trim()}` });
          i += 1;
        }
      }
      blocks.push({ type: "list", ordered: false, items });
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: InlineNode[][] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(parseInline(lines[i].replace(/^\s*\d+\.\s+/, "").trim()));
        i += 1;
      }
      blocks.push({ type: "list", ordered: true, items });
      continue;
    }

    // Blank line
    if (line.trim() === "") {
      i += 1;
      continue;
    }

    // Paragraph: gather until blank line or a new block construct
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^#{1,6}\s/.test(lines[i]) &&
      !/^```/.test(lines[i]) &&
      !/^\s*>/.test(lines[i]) &&
      !/^\s*[-*+]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i]) &&
      !/^\s*\|/.test(lines[i]) &&
      !/^\s*(---+|\*\*\*+|___+)\s*$/.test(lines[i])
    ) {
      paraLines.push(lines[i].trim());
      i += 1;
    }
    if (paraLines.length > 0) {
      blocks.push({ type: "paragraph", children: parseInline(paraLines.join(" ")) });
    }
  }

  const wordCount = cleaned
    .replace(/[#*`>|\-]+/g, " ")
    .split(/\s+/)
    .filter(Boolean).length;

  return { title, subtitle, blocks, toc, wordCount };
}

/* ------------------------------------------------------------------ */
/* Public loaders                                                      */
/* ------------------------------------------------------------------ */

function readDocFile(slug: string): string | null {
  const filePath = path.join(DOCS_DIR, `${slug}.md`);
  try {
    return fs.readFileSync(filePath, "utf-8");
  } catch {
    return null;
  }
}

export function getDoc(slug: string): Doc | null {
  const meta = DOC_REGISTRY.find((d) => d.slug === slug);
  if (!meta) return null;
  const raw = readDocFile(slug);
  if (!raw) return null;

  const parsed = parseDocument(raw);
  return {
    meta,
    title: parsed.title || meta.title,
    subtitle: parsed.subtitle,
    blocks: parsed.blocks,
    toc: parsed.toc,
    wordCount: parsed.wordCount,
    readingTimeMinutes: Math.max(1, Math.ceil(parsed.wordCount / 220)),
  };
}

export function listDocs(): DocMeta[] {
  return [...DOC_REGISTRY].sort((a, b) => a.order - b.order);
}

export function docExists(slug: string): boolean {
  return KNOWN_SLUGS.has(slug) && readDocFile(slug) !== null;
}

export function adjacentDocs(slug: string): { prev: DocMeta | null; next: DocMeta | null } {
  const docs = listDocs();
  const index = docs.findIndex((d) => d.slug === slug);
  return {
    prev: index > 0 ? docs[index - 1] : null,
    next: index >= 0 && index < docs.length - 1 ? docs[index + 1] : null,
  };
}
