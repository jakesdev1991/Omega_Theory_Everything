import Link from "next/link";

import type { DocBlock, DocHeading, InlineNode } from "@/lib/docs";

/**
 * Renders parsed C.A.R.E. Economy document blocks as React components.
 * Server component; no client JS, no raw HTML interpolation.
 */

export function renderInline(nodes: InlineNode[]): React.ReactNode[] {
  return nodes.map((node, i) => {
    switch (node.kind) {
      case "text":
        return <span key={i}>{node.text}</span>;
      case "bold":
        return <strong key={i}>{renderInline(node.children)}</strong>;
      case "italic":
        return <em key={i}>{renderInline(node.children)}</em>;
      case "code":
        return (
          <code key={i} className="doc-inline-code">
            {node.text}
          </code>
        );
      case "link":
        if (node.href.startsWith("#") || node.href.startsWith("/")) {
          return (
            <Link key={i} href={node.href} className="doc-link">
              {renderInline(node.children)}
            </Link>
          );
        }
        return (
          <a
            key={i}
            href={node.href}
            className="doc-link"
            target="_blank"
            rel="noopener noreferrer"
          >
            {renderInline(node.children)}
            {node.external ? <span className="doc-link-ext">↗</span> : null}
          </a>
        );
    }
  });
}

function renderBlock(block: DocBlock, index: number): React.ReactNode {
  switch (block.type) {
    case "paragraph":
      return <p key={index}>{renderInline(block.children)}</p>;

    case "heading": {
      if (block.level === 2) {
        return (
          <h2 key={index} id={block.id} className="doc-h2">
            {renderInline(block.children)}
          </h2>
        );
      }
      if (block.level === 3) {
        return (
          <h3 key={index} id={block.id} className="doc-h3">
            {renderInline(block.children)}
          </h3>
        );
      }
      return (
        <h4 key={index} id={block.id} className="doc-h4">
          {renderInline(block.children)}
        </h4>
      );
    }

    case "list":
      if (block.ordered) {
        return (
          <ol key={index} className="doc-ol">
            {block.items.map((item, j) => (
              <li key={j}>{renderInline(item)}</li>
            ))}
          </ol>
        );
      }
      return (
        <ul key={index} className="doc-ul">
          {block.items.map((item, j) => (
            <li key={j}>{renderInline(item)}</li>
          ))}
        </ul>
      );

    case "code":
      return (
        <pre key={index} className="doc-pre">
          <code>{block.text}</code>
        </pre>
      );

    case "blockquote":
      return (
        <blockquote key={index} className="doc-blockquote">
          {block.children.map((line, j) => (
            <p key={j}>{renderInline(line)}</p>
          ))}
        </blockquote>
      );

    case "table":
      return (
        <div key={index} className="doc-table-wrap">
          <table className="doc-table">
            <thead>
              <tr>
                {block.headers.map((cell, j) => (
                  <th key={j}>{renderInline(cell)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, j) => (
                <tr key={j}>
                  {row.map((cell, k) => (
                    <td key={k}>{renderInline(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );

    case "hr":
      return <hr key={index} className="doc-hr" />;
  }
}

export function DocReader({ blocks }: { blocks: DocBlock[] }) {
  return <div className="doc-body">{blocks.map(renderBlock)}</div>;
}

export function DocToc({ headings }: { headings: DocHeading[] }) {
  if (headings.length === 0) return null;
  return (
    <nav className="doc-toc" aria-label="Table of contents">
      <div className="doc-toc-title">Contents</div>
      <ul>
        {headings.map((h) => (
          <li key={h.id} className={h.level === 3 ? "doc-toc-sub" : undefined}>
            <a href={`#${h.id}`}>{h.text}</a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
