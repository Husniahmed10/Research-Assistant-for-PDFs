"use client";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { findCitation, safeHref } from "@/lib/conversation";
import type { Citation } from "@/lib/types";
type Node = { type: string; value?: string; url?: string; children?: Node[] };
function citationLinks() {
  return (root: unknown) => {
    function walk(node: Node) {
      if (!node.children || node.type === "link" || node.type === "linkReference") return;
      node.children = node.children.flatMap((child) => {
        if (child.type !== "text" || !child.value) {
          walk(child);
          return [child];
        }
        const result: Node[] = [];
        let start = 0;
        for (const match of child.value.matchAll(/\[(\d+)\]/g)) {
          const index = match.index!;
          if (index > start) result.push({ type: "text", value: child.value.slice(start, index) });
          result.push({
            type: "link",
            url: `#source-${match[1]}`,
            children: [{ type: "text", value: `[${match[1]}]` }],
          });
          start = index + match[0].length;
        }
        if (start < child.value.length)
          result.push({ type: "text", value: child.value.slice(start) });
        return result;
      });
    }
    walk(root as Node);
  };
}
export function Markdown({
  content,
  citations = [],
  onCitation,
}: {
  content: string;
  citations?: Citation[];
  onCitation?: (citation: Citation) => void;
}) {
  return (
    <div className="markdown">
      <ReactMarkdown
        skipHtml
        remarkPlugins={[remarkGfm, citationLinks]}
        urlTransform={safeHref}
        components={{
          a: ({ href, children }) => {
            const source = href?.match(/^#source-(\d+)$/);
            if (source) {
              const citation = findCitation(citations, source[1]);
              return citation && onCitation ? (
                <button
                  className="citation-inline"
                  aria-label={`Source ${source[1]}, page ${citation.page_num}`}
                  onClick={() => onCitation(citation)}
                >
                  {children}
                </button>
              ) : (
                <span>{children}</span>
              );
            }
            return href ? (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            ) : (
              <span>{children}</span>
            );
          },
          img: () => null,
          table: ({ children }) => (
            <div className="table-scroll">
              <table>{children}</table>
            </div>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
