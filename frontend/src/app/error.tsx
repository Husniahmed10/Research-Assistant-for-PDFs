"use client";
import { ErrorNotice } from "@/components/ui";
export default function ErrorPage({ reset }: { error: Error; reset: () => void }) {
  return (
    <div className="page">
      <h1>Unable to open this workspace</h1>
      <ErrorNotice
        error={
          new Error(
            "Something interrupted the page. Your saved papers and conversations remain on the backend.",
          )
        }
        retry={reset}
      />
    </div>
  );
}
