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
  occ_stats: Record<string, number>;
  geo_score: number | null;
  text_score: number | null;
  final_score: number;
  badge: "near-duplicate" | "weak-match" | null;
  mesh_path: string;
  thumb_path: string;
}

export type TextSource = "none" | "auto_histogram" | "user" | "user_and_histogram";

export interface SearchResponse {
  results: SearchResult[];
  query_histogram: Record<string, number>;
  query_occ_stats: Record<string, number>;
  text_source: TextSource;
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

async function _fail(res: Response): Promise<never> {
  const err = await res.json().catch(() => ({ detail: res.statusText }));
  throw new Error(err.detail ?? "Request failed");
}

/** Combined CAD + optional text query. If `file` is omitted, falls back to
 * a pure text query (no query geometry, geo_score comes back null). */
export async function search(
  params: { file?: File; text?: string; k?: number },
  onStage?: (s: SearchStage) => void
): Promise<SearchResponse> {
  const { file, text = "", k = 5 } = params;

  if (file) {
    onStage?.("reading");
    const form = new FormData();
    form.append("file", file);
    if (text.trim()) form.append("text", text.trim());
    onStage?.("graph");
    const res = await fetch(`/api/search/cad?k=${k}`, { method: "POST", body: form });
    onStage?.("searching");
    if (!res.ok) return _fail(res);
    onStage?.("done");
    return res.json();
  }

  onStage?.("searching");
  const res = await fetch("/api/search/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ q: text, k }),
  });
  if (!res.ok) return _fail(res);
  onStage?.("done");
  const data = await res.json();
  return { ...data, query_histogram: {}, query_occ_stats: {}, text_source: "user" };
}

/** Converts an uploaded STEP to a glb blob URL for the query-side 3D viewer.
 * Caller must URL.revokeObjectURL() when done. */
export async function previewMesh(file: File): Promise<string> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/mesh/preview", { method: "POST", body: form });
  if (!res.ok) return _fail(res);
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

export async function explainResult(params: {
  resultId: number;
  geoScore: number | null;
  textScore: number;
  textSource: TextSource;
  queryText?: string;
  queryHistogram?: Record<string, number>;
  queryOccStats?: Record<string, number>;
}): Promise<string> {
  const res = await fetch("/api/explain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      result_id: params.resultId,
      geo_score: params.geoScore,
      text_score: params.textScore,
      text_source: params.textSource,
      query_text: params.queryText ?? "",
      query_histogram: params.queryHistogram ?? {},
      query_occ_stats: params.queryOccStats ?? {},
    }),
  });
  if (!res.ok) return _fail(res);
  const data = await res.json();
  return data.explanation;
}

export async function askAboutResult(params: {
  resultId: number;
  question: string;
  geoScore: number | null;
  textScore: number;
  textSource: TextSource;
  queryText?: string;
  queryHistogram?: Record<string, number>;
  queryOccStats?: Record<string, number>;
}): Promise<string> {
  const res = await fetch("/api/explain/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      result_id: params.resultId,
      question: params.question,
      geo_score: params.geoScore,
      text_score: params.textScore,
      text_source: params.textSource,
      query_text: params.queryText ?? "",
      query_histogram: params.queryHistogram ?? {},
      query_occ_stats: params.queryOccStats ?? {},
    }),
  });
  if (!res.ok) return _fail(res);
  const data = await res.json();
  return data.answer;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  return res.json();
}
