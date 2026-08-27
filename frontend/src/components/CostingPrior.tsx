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
    r.geo_score !== null && `Geometry score: ${Math.round(r.geo_score * 100)}%`,
  ]
    .filter(Boolean)
    .join("\n");
}

export default function CostingPrior({ results, selected, queryHistogram }: Props) {
  // Cost band — top results with similarity >= 0.7 (geo when available, else
  // the fused/text score, so this still works for pure text-only queries).
  const qualifiedCosts = results
    .filter((r) => (r.geo_score ?? r.final_score) >= 0.7 && r.cost > 0)
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
        .filter(
          (s) =>
            s.length > 0 && Object.keys(queryHistogram).some((k) => s.toLowerCase().includes(k))
        )
    : [];

  const copyText = buildCopyText(selected, costBand ?? "", supplierHint ?? "");

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(copyText);
    } catch {
      const el = document.createElement("textarea");
      el.value = copyText;
      document.body.appendChild(el);
      el.select();
      document.execCommand("copy");
      document.body.removeChild(el);
    }
  }

  return (
    <div className="bg-ink-900/[0.03] border border-ink-400/20 rounded-xl p-4 space-y-3 text-sm">
      <h3 className="font-semibold text-ink-800 text-xs uppercase tracking-wide">
        Costing Prior
      </h3>

      {selected.cost > 0 && (
        <div className="flex justify-between">
          <span className="text-ink-400">Reference quote</span>
          <span className="font-semibold text-ink-900">${selected.cost.toLocaleString()}</span>
        </div>
      )}

      {costBand && (
        <div className="flex justify-between">
          <span className="text-ink-400">Historical range</span>
          <span className="font-medium text-ink-800">{costBand}</span>
        </div>
      )}

      {supplierHint && (
        <div className="bg-brand-50 rounded-lg px-3 py-2 text-brand-700 text-xs">
          💡 {supplierHint}
        </div>
      )}

      <div className="grid grid-cols-2 gap-2 text-xs">
        {selected.material && (
          <div>
            <span className="text-ink-400 block">Material</span>
            <span className="font-medium text-ink-800">{selected.material}</span>
          </div>
        )}
        {selected.process && (
          <div>
            <span className="text-ink-400 block">Process</span>
            <span className="font-medium text-ink-800">{selected.process}</span>
          </div>
        )}
      </div>

      {dfmWarnings.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-xs text-amber-800">
          ⚠ DFM flag from history: {dfmWarnings.join(", ")}
        </div>
      )}

      {selected.ppap_notes && (
        <div className="text-xs text-ink-400">{selected.ppap_notes}</div>
      )}

      <button
        data-testid="copy-costing-context"
        onClick={handleCopy}
        className="w-full py-2 bg-brand-500 hover:bg-brand-600 text-white text-xs font-medium rounded-lg transition-colors"
      >
        Copy context for Costing
      </button>
    </div>
  );
}
