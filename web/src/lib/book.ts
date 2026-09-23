import fs from "node:fs";
import path from "node:path";

/**
 * Chapter parser for full-book.md
 *
 * Format detected in the book:
 *   # Chapter N: <title>
 *   **POV:** ...
 *   **Setting:** ...
 *
 * We split on lines that start with "# Chapter" at the top level.
 */
export interface Chapter {
  number: number;
  title: string;
  slug: string;
  rawMd: string;
  html: string;
  wordCount: number;
  pov?: string;
  setting?: string;
  readingTimeMinutes: number;
}

const CHAPTER_TITLE_REGEX = /^#\s+Chapter\s+(\d+):\s*(.+)$/;

export function parseChapters(markdown: string): Chapter[] {
  const lines = markdown.split("\n");
  const chapters: Chapter[] = [];

  let currentChapter = "";
  let currentNumber = 0;
  let currentTitle = "";
  let started = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    const match = line.match(CHAPTER_TITLE_REGEX);
    if (match) {
      if (started && currentChapter.trim()) {
        chapters.push(
          buildChapter(
            currentNumber,
            currentTitle,
            currentChapter.trim()
          )
        );
      }

      currentNumber = parseInt(match[1], 10);
      currentTitle = match[2].trim();
      currentChapter = line + "\n";
      started = true;
      continue;
    }

    if (started) {
      currentChapter += line + "\n";
    }
  }

  if (started && currentChapter.trim()) {
    chapters.push(
      buildChapter(currentNumber, currentTitle, currentChapter.trim())
    );
  }

  return chapters;
}

function buildChapter(
  number: number,
  title: string,
  raw: string
): Chapter {
  const povMatch = raw.match(/\*\*POV:\*\*\s*(.+)/);
  const settingMatch = raw.match(/\*\*Setting:\*\*\s*(.+)/);

  // Very simple inline markdown -> HTML for reading view.
  // In production you'd use a real MDX/remark pipeline. This is
  // sufficient for a "read online" viewer because the book is
  // mostly paragraphs, chapters headings, and italic/bold inline.
  const html = renderReadingView(raw);

  const bodyText = stripMarkdownForCount(raw);
  const wordCount = bodyText.split(/\s+/).filter(Boolean).length;
  const readingTimeMinutes = Math.max(1, Math.ceil(wordCount / 220));

  return {
    number,
    title,
    slug: `chapter-${number}-${title
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/-+/g, "-")
      .slice(0, 60)}`,
    rawMd: raw,
    html,
    wordCount,
    pov: povMatch?.[1].trim() || undefined,
    setting: settingMatch?.[1].trim() || undefined,
    readingTimeMinutes,
  };
}

function stripMarkdownForCount(md: string): string {
  return md
    .replace(/^#.+$/gm, "")
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/\[(.+?)\]\(.+?\)/g, "$1")
    .replace(/^[-*+]\s+/gm, "")
    .replace(/^>\s+/gm, "")
    .trim();
}

function renderReadingView(raw: string): string {
  let out = raw;

  // Chapter title
  out = out.replace(/^#\s+Chapter\s+\d+:\s*(.+)$/gm, (_, title) => {
    return `<h2 class="chapter-title">${escapeHtml(title.trim())}</h2>`;
  });

  // Headings h3-h6
  out = out.replace(/^###\s+(.+)$/gm, (_, t) => `<h3>${escapeHtml(t.trim())}</h3>`);
  out = out.replace(/^##\s+(.+)$/gm, (_, t) => `<h3>${escapeHtml(t.trim())}</h3>`);

  // Blockquote
  out = out.replace(/^>\s+(.+)$/gm, (_, t) => {
    return `<blockquote>${escapeHtml(t.trim())}</blockquote>`;
  });

  // Unordered lists
  out = out.replace(/^[*-+]\s+(.+)$/gm, (_, t) => `<li>${inlineMarkdown(t.trim())}</li>`);

  // Paragraphs
  out = out
    .split("\n\n")
    .map((block) => {
      const blockTrimmed = block.trim();
      if (!blockTrimmed) return "";
      if (blockTrimmed.startsWith("<h2") || blockTrimmed.startsWith("<h3") ||
          blockTrimmed.startsWith("<blockquote>") || blockTrimmed.startsWith("<li>") ||
          blockTrimmed.startsWith("<hr")) {
        return blockTrimmed;
      }
      const lines = blockTrimmed.split("\n").map((l) => l.trim()).filter(Boolean);
      const content = lines.map((l) => inlineMarkdown(l)).join(" ");
      if (blockTrimmed.startsWith("|")) {
        return `<div class="callout">${content}</div>`;
      }
      return `<p>${content}</p>`;
    })
    .filter(Boolean)
    .join("\n");

  return out;
}

function inlineMarkdown(text: string): string {
  let s = escapeHtml(text);
  // Bold
  s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Italic
  s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
  // Inline code
  s = s.replace(/`([^`]+)`/g, "<code class='inline-code'>$1</code>");
  // Links
  return s;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function loadChapters(): Chapter[] {
  const filePath = path.join(process.cwd(), "public", "book", "full-book.md");
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    return parseChapters(raw);
  } catch (err) {
    console.error("Failed to load book chapters:", err);
    return [];
  }
}

export function chapterBySlug(slug: string): Chapter | undefined {
  return loadChapters().find((ch) => ch.slug === slug);
}

export function chapterByNumber(number: number): Chapter | undefined {
  return loadChapters().find((ch) => ch.number === number);
}
