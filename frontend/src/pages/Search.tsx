import { useState } from "react";
import type { SearchResult, SearchStage } from "@/api/client";
import { searchByFile, searchByText } from "@/api/client";
import ResultCard from "@/components/ResultCard";
import SearchBar from "@/components/SearchBar";

export default function Search() {
  const [stage, setStage] = useState<SearchStage>("idle");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [queryHistogram, setQueryHistogram] = useState<Record<string, number>>({});
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  async function handleFile(file: File) {
    setErrorMsg(null);
    setResults([]);
    setSelectedId(null);
    try {
      const data = await searchByFile(file, setStage);
      setResults(data.results);
      setQueryHistogram(data.query_histogram);
      setLatencyMs(data.latency_ms);
      setStage("done");
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Unknown error");
      setStage("error");
    }
  }

  async function handleText(q: string) {
    setErrorMsg(null);
    setResults([]);
    setSelectedId(null);
    setStage("searching");
    try {
      const data = await searchByText(q);
      setResults(data.results);
      setQueryHistogram({});
      setLatencyMs(data.latency_ms);
      setStage("done");
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Unknown error");
      setStage("error");
    }
  }

  const noResults = stage === "done" && results.length === 0;
  const topScoreWeak =
    results.length > 0 && results[0].final_score < 0.55;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Top bar */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-3">
        <span className="text-lg font-bold text-gray-900">CAD Search</span>
        <span className="text-xs text-gray-400">by Pre6</span>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-6 flex gap-6">
        {/* Left panel — search input */}
        <div className="w-72 shrink-0">
          <SearchBar
            stage={stage}
            onFileSubmit={handleFile}
            onTextSubmit={handleText}
          />

          {latencyMs !== null && stage === "done" && (
            <p className="mt-3 text-xs text-gray-400">
              {results.length} result{results.length !== 1 ? "s" : ""} · {latencyMs} ms
            </p>
          )}
        </div>

        {/* Right panel — results */}
        <div className="flex-1 min-w-0">
          {/* Error */}
          {errorMsg && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 text-sm text-red-700">
              {errorMsg}
            </div>
          )}

          {/* No results */}
          {noResults && (
            <div className="bg-white border border-gray-200 rounded-xl p-6 text-center">
              <p className="text-gray-500 text-sm">
                No close geometry in the library — expected for a first-of-kind
                part. Start a fresh Costing run.
              </p>
            </div>
          )}

          {/* Weak match banner */}
          {results.length > 0 && topScoreWeak && (
            <div className="mb-4 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800">
              No strong matches found. These are the closest results but scores
              are low — this may be a first-of-kind part.
            </div>
          )}

          {/* Result cards */}
          {results.length > 0 && (
            <div className="space-y-3">
              {results.map((r, i) => (
                <ResultCard
                  key={r.id}
                  result={r}
                  queryHistogram={queryHistogram}
                  rank={i + 1}
                  selected={selectedId === r.id}
                  onClick={() =>
                    setSelectedId(selectedId === r.id ? null : r.id)
                  }
                />
              ))}
            </div>
          )}

          {/* Empty state */}
          {stage === "idle" && (
            <div className="flex flex-col items-center justify-center h-64 text-gray-300">
              <div className="text-6xl mb-3">🔩</div>
              <p className="text-sm">Upload a STEP file or search by text to find similar parts</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
