"use client";
import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { api } from "@/lib/api";
import type { Paper } from "@/lib/types";
import { removePaperSessions } from "@/lib/sessions";
import { Button, ErrorNotice, Modal } from "@/components/ui";
export function DeletePaper({
  paper,
  children,
  onDeleted,
}: {
  paper: Paper;
  children: React.ReactNode;
  onDeleted?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const client = useQueryClient();
  const deletion = useMutation({
    mutationFn: () => api.delete(paper.id),
    onSuccess: async () => {
      try {
        removePaperSessions(paper.id);
      } catch {
        toast.error("Paper deleted, but browser conversation links could not be cleared.");
      }
      await client.cancelQueries();
      client.removeQueries({
        predicate: (query) => query.queryKey[0] !== "connection" && query.queryKey[0] !== "papers",
      });
      await client.invalidateQueries({ queryKey: ["papers"] });
      setOpen(false);
      toast.success("Paper deleted");
      onDeleted?.();
    },
  });
  return (
    <>
      <button
        className="icon-button delete-button"
        onClick={() => {
          deletion.reset();
          setOpen(true);
        }}
        disabled={["processing", "uploading"].includes(paper.status)}
        aria-label={`Delete ${paper.filename}`}
        title="Delete paper"
      >
        {children}
      </button>
      <Modal
        open={open}
        onOpenChange={(value) => {
          if (!deletion.isPending) setOpen(value);
        }}
        title="Delete this paper?"
        description="The PDF, indexed evidence, and its saved conversations will be permanently removed."
      >
        <p className="delete-filename">{paper.filename}</p>
        {deletion.isError && <ErrorNotice error={deletion.error} />}
        <div className="dialog-actions">
          <Button variant="secondary" disabled={deletion.isPending} onClick={() => setOpen(false)}>
            Keep paper
          </Button>
          <Button variant="danger" busy={deletion.isPending} onClick={() => deletion.mutate()}>
            Delete paper
          </Button>
        </div>
      </Modal>
    </>
  );
}
