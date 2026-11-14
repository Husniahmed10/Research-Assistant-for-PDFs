"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BookOpen, ChevronRight, Library, Menu, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { request } from "@/lib/api";
import { usePapers } from "@/lib/queries";
import { Modal } from "./ui";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [menu, setMenu] = useState(false);
  const [about, setAbout] = useState(false);
  const { setTheme, resolvedTheme } = useTheme();
  const papers = usePapers();
  const connection = useQuery({
    queryKey: ["connection"],
    queryFn: ({ signal }) => request<{ status: string }>("/", { signal }),
    refetchInterval: 60000,
    retry: false,
  });
  const navigation = (
    <>
      <Link className="brand" href="/" onClick={() => setMenu(false)}>
        <span className="brand-mark">
          <BookOpen size={21} />
        </span>
        <span>
          PaperLens<span className="brand-subtitle">RESEARCH WORKSPACE</span>
        </span>
      </Link>
      <div className="sidebar-section-label">WORKSPACE</div>
      <Link
        className={`nav-item ${pathname === "/" ? "active" : ""}`}
        href="/"
        onClick={() => setMenu(false)}
      >
        <Library size={18} />
        Paper library<span className="nav-count">{papers.data?.length ?? "—"}</span>
      </Link>
      <div className="sidebar-section-label recent-label">RECENT PAPERS</div>
      <nav aria-label="Recent papers" className="recent-papers">
        {papers.data?.slice(0, 6).map((paper) => (
          <Link
            onClick={() => setMenu(false)}
            className={`recent-paper ${pathname === `/papers/${paper.id}` ? "selected" : ""}`}
            href={`/papers/${paper.id}`}
            key={paper.id}
          >
            <span className={`paper-dot ${paper.status}`} />
            <span>{paper.filename}</span>
          </Link>
        ))}
        {!papers.data?.length && <p className="sidebar-hint">Your papers will appear here.</p>}
      </nav>
      <div className="sidebar-bottom">
        <div className="workspace-note">
          <BookOpen size={19} />
          <p>
            Research, with evidence.<span>Every insight starts with a source.</span>
          </p>
        </div>
        <button className="nav-item" onClick={() => setAbout(true)}>
          <span className="help-mark">?</span>About this workspace
          <ChevronRight size={15} className="ml-auto" />
        </button>
      </div>
    </>
  );
  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside className="sidebar" aria-label="Workspace navigation">
        {navigation}
      </aside>
      <header className="topbar">
        <div className="breadcrumb">
          <button
            className="icon-button mobile-only"
            aria-label="Open navigation"
            onClick={() => setMenu(true)}
          >
            <Menu size={20} />
          </button>
          <span>Workspace</span>
          <ChevronRight size={13} />
          <strong>{pathname === "/" ? "Paper library" : "Paper workspace"}</strong>
        </div>
        <div className="topbar-actions">
          <span
            className={`connection ${connection.isError ? "offline" : ""}`}
            title={
              connection.isError
                ? "Check the backend connection"
                : "Lightweight API connection check"
            }
          >
            <span />
            {connection.isPending
              ? "Connecting"
              : connection.isError
                ? "API offline"
                : "API connected"}
          </span>
          <span className="topbar-divider" />
          <button
            className="icon-button theme-button"
            aria-label="Toggle color theme"
            onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
          >
            <Sun className="light-icon" size={18} />
            <Moon className="dark-icon" size={18} />
          </button>
        </div>
      </header>
      <main id="main-content" className="main-content">
        {children}
      </main>
      <Modal
        open={menu}
        onOpenChange={setMenu}
        title="Workspace navigation"
        description="Your paper library and recent documents."
        className="navigation-dialog"
      >
        <div className="mobile-navigation">{navigation}</div>
      </Modal>
      <Modal
        open={about}
        onOpenChange={setAbout}
        title="A clearer view of your research"
        description="PaperLens connects your papers, conversations, and supporting evidence."
      >
        <div className="about-copy">
          <p>
            Upload a PDF, explore its structured summary, and ask questions grounded in the paper.
          </p>
          <p>
            Conversations are saved by the research API. The list of conversation links is stored in
            this browser. Automated quality scores are estimates and should be reviewed alongside
            the evidence.
          </p>
        </div>
      </Modal>
    </div>
  );
}
