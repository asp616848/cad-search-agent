export interface SearchResult {
  id: number;
  name: string;
  material: string;
  process: string;
  cost: number;
  supplier: string;
  notes: string;
  known_issues: string;
  ppap_notes: string;
  histogram: Record<string, number>;
  geo_score: number;
  text_score: number;
  final_score: number;
  badge: "near-duplicate" | "weak-match" | null;
  mesh_path: string;
  thumb_path: string;
}

export interface CadSearchResponse {
  results: SearchResult[];
  query_histogram: Record<string, number>;
  latency_ms: number;
}

export interface TextSearchResponse {
  results: SearchResult[];
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  index_size: number;
  started_at: string;
}

export type SearchStage =
  | "idle"
  | "reading"
  | "graph"
  | "uvnet"
  | "searching"
  | "done"
  | "error";

export async function searchByFile(
  file: File,
  onStage: (s: SearchStage) => void
): Promise<CadSearchResponse> {
  onStage("reading");
  const form = new FormData();
  form.append("file", file);
  onStage("graph");
  const res = await fetch("/api/search/cad", { method: "POST", body: form });
  onStage("searching");
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Search failed");
  }
  onStage("done");
  return res.json();
}

export async function searchByText(q: string): Promise<TextSearchResponse> {
  const res = await fetch("/api/search/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Search failed");
  }
  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  return res.json();
}
