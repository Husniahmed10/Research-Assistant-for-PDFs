import Link from "next/link";
export default function NotFound() {
  return (
    <div className="page">
      <h1>Page not found</h1>
      <p className="muted">Return to your library to choose a paper.</p>
      <Link href="/" className="button primary">
        Open library
      </Link>
    </div>
  );
}
