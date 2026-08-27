import { useEffect, useRef, useState } from "react";
import type { SearchResult, SearchStage, TextSource } from "@/api/client";
import { previewMesh, search } from "@/api/client";
import CostingPrior from "@/components/CostingPrior";
import ExplainPanel from "@/components/ExplainPanel";
import ResultCard from "@/components/ResultCard";
import SearchBar from "@/components/SearchBar";
import Viewer3D from "@/components/Viewer3D";

export default function Search() {
  const [stage, setStage] = useState<SearchStage>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [queryHistogram, setQueryHistogram] = useState<Record<string, number>>({});
  const [queryOccStats, setQueryOccStats] = useState<Record<string, number>>({});
  const [textSource, setTextSource] = useState<TextSource>("none");
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [queryMeshUrl, setQueryMeshUrl] = useState<string | null>(null);
  const queryMeshUrlRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (queryMeshUrlRef.current) URL.revokeObjectURL(queryMeshUrlRef.current);
    };
  }, []);

  function setQueryMesh(url: string | null) {
    if (queryMeshUrlRef.current) URL.revokeObjectURL(queryMeshUrlRef.current);
    queryMeshUrlRef.current = url;
    setQueryMeshUrl(url);
  }

  async function handleSubmit() {
    if (!file && !text.trim()) return;
    setErrorMsg(null);
    setResults([]);
    setSelectedId(null);
    setQueryMesh(null);

    if (file) {
      previewMesh(file)
        .then(setQueryMesh)
        .catch(() => setQueryMesh(null));
    }

    try {
      const data = await search({ file: file ?? undefined, text, k: 5 }, setStage);
      setResults(data.results);
      setQueryHistogram(data.query_histogram);
      setQueryOccStats(data.query_occ_stats);
      setTextSource(data.text_source);
      setLatencyMs(data.latency_ms);
      setStage("done");
    } catch (e: unknown) {
      setErrorMsg(e instanceof Error ? e.message : "Unknown error");
      setStage("error");
    }
  }

  const noResults = stage === "done" && results.length === 0;
  const topScoreWeak = results.length > 0 && results[0].final_score < 0.55;

  return (
    <div className="min-h-screen bg-ink-900/[0.015]">
      {/* Top bar */}
      <header className="bg-white border-b border-ink-400/15 px-6 py-3.5 flex items-center gap-2.5">
        <div className="w-7 h-7 rounded-full bg-brand-500 flex items-center justify-center text-white font-bold text-sm">
          C
        </div>
        <span className="text-lg font-bold text-ink-900">CAD Search</span>
        <span className="text-xs text-ink-400">by Pre6</span>
      </header>

      <div className="max-w-6xl mx-auto px-4 py-6 flex gap-6">
        {/* Left panel — search input */}
        <div className="w-72 shrink-0">
          <SearchBar
            stage={stage}
            file={file}
            text={text}
            onFileChange={setFile}
            onTextChange={setText}
            onSubmit={handleSubmit}
          />

          {latencyMs !== null && stage === "done" && (
            <p className="mt-3 text-xs text-ink-400">
              {results.length} result{results.length !== 1 ? "s" : ""} · {latencyMs} ms
            </p>
          )}
        </div>

        {/* Right panel — results */}
        <div className="flex-1 min-w-0">
          {errorMsg && (
            <div className="bg-brand-50 border border-brand-100 rounded-xl p-4 text-sm text-brand-700">
              {errorMsg}
            </div>
          )}

          {noResults && (
            <div className="bg-white border border-ink-400/20 rounded-xl p-6 text-center">
              <p className="text-ink-400 text-sm">
                No close geometry in the library — expected for a first-of-kind part. Start a
                fresh Costing run.
              </p>
            </div>
          )}

          {results.length > 0 && topScoreWeak && (
            <div className="mb-4 bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-sm text-amber-800">
              No strong matches found. These are the closest results but scores are low — this
              may be a first-of-kind part.
            </div>
          )}

          {results.length > 0 && (
            <div className="space-y-3">
              {results.map((r, i) => {
                const isSelected = selectedId === r.id;
                const resultMeshUrl = `/api/mesh/${r.id}`;
                return (
                  <div key={r.id}>
                    <ResultCard
                      result={r}
                      queryHistogram={queryHistogram}
                      textSource={textSource}
                      rank={i + 1}
                      selected={isSelected}
                      onClick={() => setSelectedId(isSelected ? null : r.id)}
                    />
                    {isSelected && (
                      <div className="mt-2 space-y-4 border border-ink-400/20 rounded-xl p-4 bg-white">
                        {r.badge === "near-duplicate" && r.geo_score !== null && (
                          <div className="bg-brand-50 border border-brand-100 rounded-lg px-4 py-2 text-sm text-brand-700">
                            This looks like an existing part — geo similarity{" "}
                            {Math.round(r.geo_score * 100)}%. Check the library before creating a
                            new quote.
                          </div>
                        )}

                        <Viewer3D
                          queryGltfUrl={queryMeshUrl ?? undefined}
                          resultGltfUrl={resultMeshUrl}
                          resultName={r.name}
                        />

                        <div className="grid grid-cols-2 gap-4">
                          <CostingPrior
                            results={results}
                            selected={r}
                            queryHistogram={queryHistogram}
                          />
                          <ExplainPanel
                            result={r}
                            textSource={textSource}
                            queryText={text}
                            queryHistogram={queryHistogram}
                            queryOccStats={queryOccStats}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {stage === "idle" && (
            <div className="flex flex-col items-center justify-center h-64 text-ink-400/50">
              <div className="text-6xl mb-3">🔩</div>
              <p className="text-sm">Upload a STEP file or search by text to find similar parts</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
