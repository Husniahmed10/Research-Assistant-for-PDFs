"use client";
import { useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { FileText, UploadCloud } from "lucide-react";
import { toast } from "sonner";
import { api, errorMessage, MAX_FILE_SIZE_MB, validateUpload } from "@/lib/api";
import { Button, Modal } from "@/components/ui";
export function UploadDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const lock = useRef(false);
  const client = useQueryClient();
  const upload = useMutation({
    mutationFn: api.upload,
    onSuccess: async () => {
      await client.invalidateQueries({ queryKey: ["papers"] });
      toast.success("PDF uploaded. Your paper is now processing.");
      onOpenChange(false);
      setFile(null);
    },
    onError: (error) => setError(errorMessage(error)),
  });
  function choose(files: FileList | null) {
    if (!files?.length) return;
    if (files.length > 1) {
      setFile(null);
      setError("Please upload one PDF at a time.");
      return;
    }
    const issue = validateUpload(files[0]);
    setError(issue || "");
    setFile(issue ? null : files[0]);
  }
  return (
    <Modal
      open={open}
      onOpenChange={(value) => {
        if (!lock.current) {
          onOpenChange(value);
          setError("");
        }
      }}
      title="Add a paper to your library"
      description="Upload a research PDF. We’ll extract its text, create a summary, and prepare it for questions."
    >
      <form
        onSubmit={async (event) => {
          event.preventDefault();
          if (!file || lock.current) return;
          lock.current = true;
          try {
            await upload.mutateAsync(file);
          } catch {
            /* Mutation displays the error. */
          } finally {
            lock.current = false;
          }
        }}
      >
        <input
          ref={input}
          type="file"
          accept="application/pdf,.pdf"
          aria-label="Choose PDF"
          className="sr-only"
          disabled={upload.isPending}
          onChange={(event) => {
            choose(event.target.files);
            event.target.value = "";
          }}
        />
        <button
          type="button"
          disabled={upload.isPending}
          onClick={() => input.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            if (!lock.current) choose(event.dataTransfer.files);
          }}
          className={`dropzone ${dragging ? "dragging" : ""}`}
        >
          <span className="dropzone-icon">
            {file ? <FileText size={28} /> : <UploadCloud size={28} />}
          </span>
          <strong>{file ? file.name : "Drop your PDF here"}</strong>
          <span>
            {file
              ? `${(file.size / 1024 / 1024).toFixed(2)} MB · Click to replace`
              : "or click to browse files"}
          </span>
          <small>PDF only · Up to {MAX_FILE_SIZE_MB} MB</small>
        </button>
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        <p className="upload-note">
          Processing continues after upload. Your paper will appear in the library while it’s being
          prepared.
        </p>
        <div className="dialog-actions">
          <Button
            type="button"
            variant="secondary"
            disabled={upload.isPending}
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button type="submit" disabled={!file} busy={upload.isPending}>
            {upload.isPending ? "Uploading PDF…" : "Upload paper"}
          </Button>
        </div>
      </form>
    </Modal>
  );
}
