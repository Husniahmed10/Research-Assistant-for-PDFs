import { PaperWorkspace } from "@/features/paper/paper-workspace";
export default async function PaperPage({ params }: { params: Promise<{ paperId: string }> }) {
  const { paperId } = await params;
  return <PaperWorkspace key={paperId} paperId={paperId} />;
}
