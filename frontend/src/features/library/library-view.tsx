"use client";
import { useState } from "react";
import Link from "next/link";
import {
  ArrowDownUp,
  ArrowUpRight,
  FileText,
  Plus,
  Search,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { usePapers } from "@/lib/queries";
import { Button, Empty, ErrorNotice, Skeleton, Status } from "@/components/ui";
import { UploadDialog } from "./upload-dialog";
import { DeletePaper } from "./delete-paper";
export function LibraryView() {
  const papers = usePapers();
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [oldest, setOldest] = useState(false);
  const [upload, setUpload] = useState(false);
  const filtered = (papers.data || [])
    .filter(
      (paper) =>
        paper.filename.toLowerCase().includes(search.toLowerCase()) &&
        (filter === "all" || paper.status === filter),
    )
    .sort((a, b) => (oldest ? 1 : -1) * (Date.parse(a.created_at) - Date.parse(b.created_at)));
  return (
    <div className="page library-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">YOUR RESEARCH, ORGANIZED</div>
          <h1>
            Paper library<span className="title-period">.</span>
          </h1>
          <p>A home for your papers. A starting point for your next insight.</p>
        </div>
        <Button onClick={() => setUpload(true)}>
          <Plus size={17} />
          Upload paper
        </Button>
      </div>
      <section className="library-intro">
        <div className="intro-graphic" aria-hidden="true">
          <FileText size={33} strokeWidth={1.3} />
          <span className="graphic-tag">PDF</span>
        </div>
        <div>
          <span className="eyebrow">FROM READING TO UNDERSTANDING</span>
          <h2>
            Spend less time searching.
            <br />
            More time connecting ideas.
          </h2>
          <p>
            Bring a paper. Explore its summary, ask questions,
            <br className="desktop-break" /> and trace every cited insight back to the evidence.
          </p>
        </div>
        <div className="intro-side">
          <span>
            01 <b>Upload a paper</b>
          </span>
          <span>
            02 <b>Explore the essentials</b>
          </span>
          <span>
            03 <b>Ask. Verify. Discover.</b>
          </span>
        </div>
      </section>
      <div className="section-heading">
        <h2>
          Your papers <span className="count-pill">{papers.data?.length ?? "—"}</span>
        </h2>
        <span className="muted small">
          {papers.isFetching && !papers.isPending
            ? "Updating library…"
            : "Your research collection"}
        </span>
      </div>
      <div className="library-toolbar">
        <label className="search-field">
          <Search size={17} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search papers by filename…"
            aria-label="Search papers"
          />
        </label>
        <label className="filter-label">
          <span className="sr-only">Filter by status</span>
          <select value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="all">All statuses</option>
            <option value="ready">Ready</option>
            <option value="processing">Processing</option>
            <option value="failed">Failed</option>
          </select>
        </label>
        <Button
          variant="secondary"
          className="sort-button"
          onClick={() => setOldest(!oldest)}
          aria-label={oldest ? "Sort newest first" : "Sort oldest first"}
        >
          <ArrowDownUp size={15} />
          <span>{oldest ? "Oldest first" : "Newest first"}</span>
        </Button>
      </div>
      {papers.isError && <ErrorNotice error={papers.error} retry={() => void papers.refetch()} />}
      {papers.isPending ? (
        <Skeleton rows={4} />
      ) : filtered.length ? (
        <div className="paper-table">
          <div className="paper-table-head">
            <span>DOCUMENT</span>
            <span>STATUS</span>
            <span>ADDED</span>
            <span />
          </div>
          {filtered.map((paper) => (
            <div className="paper-row" key={paper.id}>
              <Link className="paper-identity" href={`/papers/${paper.id}`}>
                <span className="pdf-icon">
                  <FileText size={22} strokeWidth={1.5} />
                  <small>PDF</small>
                </span>
                <span>
                  <strong>{paper.filename}</strong>
                  <span className="paper-meta">
                    {paper.page_count === null ? "Pages pending" : `${paper.page_count} pages`}
                    <span>·</span>
                    {paper.file_size_kb >= 1024
                      ? `${(paper.file_size_kb / 1024).toFixed(1)} MB`
                      : `${Math.round(paper.file_size_kb)} KB`}
                  </span>
                </span>
              </Link>
              <Status status={paper.status} />
              <time className="paper-date" dateTime={paper.created_at}>
                {new Date(paper.created_at).toLocaleDateString(undefined, {
                  month: "short",
                  day: "numeric",
                  year: "numeric",
                })}
              </time>
              <div className="row-actions">
                <DeletePaper paper={paper}>
                  <Trash2 size={16} />
                </DeletePaper>
                <Link
                  className="icon-button"
                  href={`/papers/${paper.id}`}
                  aria-label={`Open ${paper.filename}`}
                >
                  <ArrowUpRight size={19} />
                </Link>
              </div>
            </div>
          ))}
        </div>
      ) : (
        !papers.isError && (
          <Empty
            title={papers.data?.length ? "No matching papers" : "Your next insight starts here"}
            action={
              <Button
                variant="secondary"
                onClick={() =>
                  papers.data?.length ? (setSearch(""), setFilter("all")) : setUpload(true)
                }
              >
                {papers.data?.length ? (
                  "Clear filters"
                ) : (
                  <>
                    <UploadCloud size={16} />
                    Upload your first paper
                  </>
                )}
              </Button>
            }
          >
            {papers.data?.length
              ? "Try another filename or choose a different status."
              : "Add a research PDF to get a structured summary and start a conversation with the evidence."}
          </Empty>
        )
      )}
      <div className="library-footer">
        <span>Grounded in your papers. Built for thoughtful research.</span>
        <span>PDF documents</span>
      </div>
      <UploadDialog open={upload} onOpenChange={setUpload} />
    </div>
  );
}
