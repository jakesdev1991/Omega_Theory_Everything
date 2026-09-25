// Copyright (c) 2025-2026 Jacob See.
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import test from "node:test";
import assert from "node:assert/strict";

import { LEGAL_DOCUMENTS, loadLegalDocument, renderLegalMarkdown } from "./legal";
import { PUBLISHER_AGREEMENT_ID, STORE_TERMS_ID } from "./store-license";

test("Legal renderer escapes HTML and neutralizes unsafe links", () => {
  const html = renderLegalMarkdown('Hello <script>alert(1)</script> [x](javascript:alert(1)) [ok](https://example.com) **b** `c`');
  assert.ok(!html.includes("<script>"));
  assert.ok(html.includes("&lt;script&gt;"));
  assert.ok(html.includes('href="#"'), "javascript: link neutralized");
  assert.ok(html.includes('href="https://example.com"'));
  assert.ok(html.includes("<strong>b</strong>") && html.includes("<code>c</code>"));
});

test("Legal renderer handles headings, lists, quotes and tables", () => {
  const html = renderLegalMarkdown("# T\n\n> note\n\n1. a\n2. b\n\n- x\n- y\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\npara\nline");
  assert.match(html, /<h1 id="t">T<\/h1>/);
  assert.match(html, /<blockquote>/);
  assert.match(html, /<ol><li>a<\/li><li>b<\/li><\/ol>/);
  assert.match(html, /<ul><li>x<\/li><li>y<\/li><\/ul>/);
  assert.match(html, /<th>A<\/th>/);
  assert.match(html, /<p>para line<\/p>/);
});

test("Shipped legal documents load and carry the ids the code references", () => {
  for (const { file } of LEGAL_DOCUMENTS) {
    const doc = loadLegalDocument(file);
    assert.ok(doc.html.length > 1000, `${file} renders`);
  }
  assert.ok(loadLegalDocument("store-terms.md").raw.includes(STORE_TERMS_ID));
  assert.ok(loadLegalDocument("publisher-agreement.md").raw.includes(PUBLISHER_AGREEMENT_ID));
});
