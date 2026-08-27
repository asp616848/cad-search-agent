import { useState } from "react";
import type { SearchResult } from "@/api/client";

interface Props {
  result: SearchResult;
  queryHistogram?: Record<string, number>;
  rank: number;
  onClick?: () => void;
  selected?: boolean;
}

function ScoreBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number | null;
  color: string;
}) {
  if (value === null) {
    return (
      <div className="flex items-center gap-2 text-xs">
        <span className="w-16 text-ink-400 shrink-0">{label}</span>
        <div className="flex-1 bg-ink-400/10 rounded-full h-1.5" />
        <span className="w-8 text-right font-mono text-ink-400">N/A</span>
      </div>
    );
  }
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 text-ink-400 shrink-0">{label}</span>
      <div className="flex-1 bg-ink-400/10 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 text-right font-mono text-ink-800">{pct}%</span>
    </div>
  );
}

function FeatureChip({
  label,
  count,
  overlap,
}: {
  label: string;
  count: number;
  overlap: boolean;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
        overlap ? "bg-green-100 text-green-800" : "bg-ink-400/10 text-ink-400"
      }`}
    >
      {label}
      <span className="font-mono">{count}</span>
    </span>
  );
}

function Thumbnail({ resultId }: { resultId: number }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div className="w-16 h-16 rounded-lg bg-ink-900/[0.04] flex items-center justify-center text-lg shrink-0">
        📐
      </div>
    );
  }
  return (
    <img
      src={`/api/thumbnail/${resultId}`}
      alt=""
      className="w-16 h-16 rounded-lg object-cover shrink-0 bg-ink-900"
      onError={() => setFailed(true)}
    />
  );
}

export default function ResultCard({
  result,
  queryHistogram,
  rank,
  onClick,
  selected,
}: Props) {
  const badge = result.badge;
  const histEntries = Object.entries(result.histogram).filter(([, v]) => v > 0);

  return (
    <div
      className={`bg-white rounded-xl border p-4 cursor-pointer transition-shadow hover:shadow-md ${
        selected ? "border-brand-500 shadow-md" : "border-ink-400/20"
      }`}
      onClick={onClick}
    >
      <div className="flex gap-3">
        <Thumbnail resultId={result.id} />

        <div className="flex-1 min-w-0">
          {/* Header row */}
          <div className="flex items-start justify-between gap-2 mb-3">
            <div className="flex items-center gap-2 min-w-0">
              <span className="text-xs font-mono text-ink-400 shrink-0">#{rank}</span>
              <span className="font-semibold text-ink-900 truncate">{result.name}</span>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              {badge === "near-duplicate" && (
                <span className="px-2 py-0.5 bg-brand-100 text-brand-700 text-xs font-medium rounded-full">
                  Near-duplicate
                </span>
              )}
              {badge === "weak-match" && (
                <span className="px-2 py-0.5 bg-ink-400/10 text-ink-600 text-xs font-medium rounded-full">
                  Weak match
                </span>
              )}
              {result.cost > 0 && (
                <span className="text-sm font-semibold text-ink-800">
                  ${result.cost.toLocaleString()}
                </span>
              )}
            </div>
          </div>

          {/* Meta pills */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {result.material && (
              <span className="px-2 py-0.5 bg-ink-900/5 text-ink-800 text-xs rounded">
                {result.material}
              </span>
            )}
            {result.process && (
              <span className="px-2 py-0.5 bg-brand-50 text-brand-700 text-xs rounded">
                {result.process}
              </span>
            )}
            {result.supplier && (
              <span className="px-2 py-0.5 bg-ink-400/10 text-ink-600 text-xs rounded">
                {result.supplier}
              </span>
            )}
          </div>

          {/* Dual score bars */}
          <div className="space-y-1.5 mb-3">
            <ScoreBar label="Geometry" value={result.geo_score} color="bg-ink-800" />
            <ScoreBar label="Text" value={result.text_score} color="bg-brand-500" />
          </div>

          {/* Feature chips */}
          {histEntries.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {histEntries.map(([label, count]) => (
                <FeatureChip
                  key={label}
                  label={label}
                  count={count}
                  overlap={!!queryHistogram && (queryHistogram[label] ?? 0) > 0}
                />
              ))}
            </div>
          )}

          {/* Known issues */}
          {result.known_issues && (
            <p className="mt-2 text-xs text-amber-700 bg-amber-50 rounded px-2 py-1">
              ⚠ {result.known_issues}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
