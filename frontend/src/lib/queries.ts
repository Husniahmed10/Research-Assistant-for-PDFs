"use client";
import { useQuery } from "@tanstack/react-query";
import { api, ApiError } from "./api";
export function usePapers() {
  return useQuery({
    queryKey: ["papers"],
    queryFn: ({ signal }) => api.papers(signal),
    refetchInterval: (query) =>
      query.state.data?.some((paper) => ["uploading", "processing"].includes(paper.status))
        ? 2500
        : false,
  });
}
export function useMessages(session?: string) {
  return useQuery({
    queryKey: ["messages", session],
    queryFn: ({ signal }) => api.messages(session!, signal),
    enabled: !!session,
    refetchOnWindowFocus: false,
  });
}
export const retryRead = (failures: number, error: Error) =>
  !(error instanceof ApiError && error.status >= 400 && error.status < 500) && failures < 1;
