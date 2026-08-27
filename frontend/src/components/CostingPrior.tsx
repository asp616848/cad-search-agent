import type { SearchResult } from "@/api/client";

interface Props {
  results: SearchResult[];
  selected: SearchResult;
  queryHistogram: Record<string, number>;
}

function buildCopyText(r: SearchResult, band: string, supplierHint: string): string {
  return [
    `Part: ${r.name}`,
    r.material && `Material: ${r.material}`,
    r.process && `Process: ${r.process}`,
    r.cost > 0 && `Reference cost: $${r.cost.toLocaleString()}`,
    band && `Cost band (similar parts): ${band}`,
    supplierHint && `Supplier hint: ${supplierHint}`,
    r.notes && `Notes: ${r.notes}`,
    r.known_issues && `Known issues: ${r.known_issues}`,
    r.ppap_notes && `PPAP: ${r.ppap_notes}`,
    `Geometry score: ${Math.round(r.geo_score * 100)}%`,
  ]
    .filter(Boolean)
    .join("\n");
}

export default function CostingPrior({ results, selected, queryHistogram }: Props) {
  // Cost band — top results with geo >= 0.7
  const qualifiedCosts = results
    .filter((r) => r.geo_score >= 0.7 && r.cost > 0)
    .map((r) => r.cost);
  const costBand =
    qualifiedCosts.length >= 2
      ? `$${Math.min(...qualifiedCosts).toLocaleString()}–$${Math.max(...qualifiedCosts).toLocaleString()} (n=${qualifiedCosts.length})`
      : qualifiedCosts.length === 1
      ? `$${qualifiedCosts[0].toLocaleString()} (n=1)`
      : null;

  // Supplier hint — most common supplier in top-3
  const top3 = results.slice(0, 3);
  const supplierCounts = top3.reduce<Record<string, number>>((acc, r) => {
    if (r.supplier) acc[r.supplier] = (acc[r.supplier] ?? 0) + 1;
    return acc;
  }, {});
  const topSupplier = Object.entries(supplierCounts).sort((a, b) => b[1] - a[1])[0];
  const supplierHint = topSupplier
    ? `${topSupplier[1]} of ${top3.length} closest parts went to ${topSupplier[0]}`
    : null;

  // DFM from history — known_issues token overlaps with query histogram label
  const dfmWarnings = selected.known_issues
    ? selected.known_issues
        .split(/[,;.]/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0 && Object.keys(queryHistogram).some((k) => s.toLowerCase().includes(k)))
    : [];

  const copyText = buildCopyText(selected, costBand ?? "", supplierHint ?? "");

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(copyText);
    } catch {
      // fallback for non-secure contexts
      const el = document.createElement("textarea");
      el.value = copyText;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
  }

  return (
    <div className="bg-gray-50 border border-gray-200 rounded-xl p-4 space-y-3 text-sm">
      <h3 className="font-semibold text-gray-800 text-xs uppercase tracking-wide">
        Costing Prior
      </h3>

      {/* Reference cost */}
      {selected.cost > 0 && (
        <div className="flex justify-between">
          <span className="text-gray-500">Reference quote</span>
          <span className="font-semibold text-gray-900">${selected.cost.toLocaleString()}</span>
        </div>
      )}

      {/* Cost band */}
      {costBand && (
        <div className="flex justify-between">
          <span className="text-gray-500">Historical range</span>
          <span className="font-medium text-gray-700">{costBand}</span>
        </div>
      )}

      {/* Supplier hint */}
      {supplierHint && (
        <div className="bg-blue-50 rounded-lg px-3 py-2 text-blue-800 text-xs">
          💡 {supplierHint}
        </div>
      )}

      {/* Process + material */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        {selected.material && (
          <div>
            <span className="text-gray-400 block">Material</span>
            <span className="font-medium text-gray-700">{selected.material}</span>
          </div>
        )}
        {selected.process && (
          <div>
            <span className="text-gray-400 block">Process</span>
            <span className="font-medium text-gray-700">{selected.process}</span>
          </div>
        )}
      </div>

      {/* DFM warning */}
      {dfmWarnings.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-800">
          ⚠ DFM flag from history: {dfmWarnings.join(", ")}
        </div>
      )}

      {/* PPAP */}
      {selected.ppap_notes && (
        <div className="text-xs text-gray-500">{selected.ppap_notes}</div>
      )}

      {/* Copy button */}
      <button
        data-testid="copy-costing-context"
        onClick={handleCopy}
        className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded-lg transition-colors"
      >
        Copy context for Costing
      </button>
    </div>
  );
}
