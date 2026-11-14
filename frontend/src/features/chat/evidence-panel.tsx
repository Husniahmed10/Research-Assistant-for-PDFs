"use client";
import * as Dialog from "@radix-ui/react-dialog";
import { Copy, FileText, X } from "lucide-react";
import { toast } from "sonner";
import { useRef, useSyncExternalStore } from "react";
import type { Citation } from "@/lib/types";
import { Button } from "@/components/ui";
export async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  } catch {
    toast.error("Clipboard is unavailable. You can select and copy the text manually.");
  }
}
export function EvidencePanel({
  citation,
  onClose,
}: {
  citation: Citation | null;
  onClose: () => void;
}) {
  const previousFocus = useRef<HTMLElement | null>(null);
  const wide = useSyncExternalStore(
    subscribeViewport,
    () => window.matchMedia("(min-width: 1200px)").matches,
    () => false,
  );
  return (
    <Dialog.Root
      modal={!wide}
      open={!!citation}
      onOpenChange={(value) => {
        if (!value) onClose();
      }}
    >
      <Dialog.Portal>
        {!wide && <Dialog.Overlay className="evidence-overlay" />}
        <Dialog.Content
          className="evidence-panel"
          onInteractOutside={(event) => {
            if (wide) event.preventDefault();
          }}
          onOpenAutoFocus={() => {
            previousFocus.current =
              document.activeElement instanceof HTMLElement ? document.activeElement : null;
          }}
          onCloseAutoFocus={(event) => {
            if (previousFocus.current?.isConnected) {
              event.preventDefault();
              previousFocus.current.focus();
            }
          }}
        >
          <div className="evidence-heading">
            <span className="eyebrow">FOLLOW THE EVIDENCE</span>
            <Dialog.Close className="icon-button" aria-label="Close evidence">
              <X size={18} />
            </Dialog.Close>
          </div>
          <Dialog.Title className="evidence-title">Source excerpt</Dialog.Title>
          <Dialog.Description className="evidence-description">
            Text retrieved from your paper, with its original page reference.
          </Dialog.Description>
          {citation && (
            <>
              <div className="evidence-document">
                <FileText size={18} />
                <span>Page {citation.page_num}</span>
                <span className="source-label">
                  {citation.source_ids.map((id) => `[${id}]`).join(" ")}
                </span>
              </div>
              <blockquote className="evidence-text">{citation.text}</blockquote>
              <Button variant="secondary" onClick={() => void copyText(citation.text)}>
                <Copy size={15} />
                Copy excerpt
              </Button>
              <p className="evidence-footnote">
                Check this passage in the original PDF for its full context.
              </p>
            </>
          )}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
function subscribeViewport(listener: () => void) {
  const media = window.matchMedia("(min-width: 1200px)");
  media.addEventListener("change", listener);
  return () => media.removeEventListener("change", listener);
}
