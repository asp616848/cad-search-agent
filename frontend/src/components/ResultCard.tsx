import type { SearchResult } from "@/api/client";

interface Props {
  result: SearchResult;
  queryHistogram?: Record<string, number>;
  rank: number;
  onClick?: () => void;
  selected?: boolean;
}

function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-16 text-gray-500 shrink-0">{label}</span>
      <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
        <div
          className={`h-full rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-8 text-right font-mono text-gray-700">{pct}%</span>
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
        overlap
          ? "bg-green-100 text-green-800"
          : "bg-gray-100 text-gray-500"
      }`}
    >
      {label}
      <span className="font-mono">{count}</span>
    </span>
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
        selected ? "border-blue-500 shadow-md" : "border-gray-200"
      }`}
      onClick={onClick}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-xs font-mono text-gray-400 shrink-0">#{rank}</span>
          <span className="font-semibold text-gray-900 truncate">{result.name}</span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          {badge === "near-duplicate" && (
            <span className="px-2 py-0.5 bg-orange-100 text-orange-700 text-xs font-medium rounded-full">
              Near-duplicate
            </span>
          )}
          {badge === "weak-match" && (
            <span className="px-2 py-0.5 bg-gray-100 text-gray-500 text-xs font-medium rounded-full">
              Weak match
            </span>
          )}
          {result.cost > 0 && (
            <span className="text-sm font-semibold text-gray-700">
              ${result.cost.toLocaleString()}
            </span>
          )}
        </div>
      </div>

      {/* Meta pills */}
      <div className="flex flex-wrap gap-1.5 mb-3">
        {result.material && (
          <span className="px-2 py-0.5 bg-blue-50 text-blue-700 text-xs rounded">
            {result.material}
          </span>
        )}
        {result.process && (
          <span className="px-2 py-0.5 bg-purple-50 text-purple-700 text-xs rounded">
            {result.process}
          </span>
        )}
        {result.supplier && (
          <span className="px-2 py-0.5 bg-gray-50 text-gray-600 text-xs rounded">
            {result.supplier}
          </span>
        )}
      </div>

      {/* Dual score bars */}
      <div className="space-y-1.5 mb-3">
        <ScoreBar label="Geometry" value={result.geo_score} color="bg-blue-500" />
        <ScoreBar label="Text" value={result.text_score} color="bg-violet-400" />
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
  );
}
