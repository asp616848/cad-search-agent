import { useEffect, useState } from "react";
import type { SearchResult, TextSource } from "@/api/client";
import { askAboutResult, explainResult } from "@/api/client";

interface Props {
  result: SearchResult;
  textSource: TextSource;
  queryText: string;
  queryHistogram: Record<string, number>;
  queryOccStats: Record<string, number>;
}

export default function ExplainPanel({
  result,
  textSource,
  queryText,
  queryHistogram,
  queryOccStats,
}: Props) {
  const [explanation, setExplanation] = useState<string | null>(null);
  const [loadingExplain, setLoadingExplain] = useState(true);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setExplanation(null);
    setAnswer(null);
    setLoadingExplain(true);
    explainResult({
      resultId: result.id,
      geoScore: result.geo_score,
      textScore: result.text_score ?? 0,
      textSource,
      queryText,
      queryHistogram,
      queryOccStats,
    })
      .then((text) => {
        if (!cancelled) setExplanation(text);
      })
      .catch(() => {
        if (!cancelled) setExplanation(null);
      })
      .finally(() => {
        if (!cancelled) setLoadingExplain(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result.id, result.geo_score, result.text_score, textSource, queryText]);

  async function handleAsk() {
    if (!question.trim()) return;
    setAsking(true);
    setAnswer(null);
    try {
      const a = await askAboutResult({
        resultId: result.id,
        question: question.trim(),
        geoScore: result.geo_score,
        textScore: result.text_score ?? 0,
        textSource,
        queryText,
        queryHistogram,
        queryOccStats,
      });
      setAnswer(a);
    } catch (e: unknown) {
      setAnswer(e instanceof Error ? e.message : "Could not get an answer.");
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="bg-white border border-ink-400/20 rounded-xl p-4 space-y-3 text-sm">
      <h3 className="font-semibold text-ink-800 text-xs uppercase tracking-wide">
        Why this match
      </h3>

      {textSource === "auto_histogram" && (
        <p className="text-ink-400 text-xs italic">
          No text was typed — the Text score compares this file's own detected features
          against each result's description.
        </p>
      )}

      {loadingExplain && (
        <p className="text-ink-400 text-xs italic">Generating explanation…</p>
      )}
      {!loadingExplain && explanation && (
        <p className="text-ink-800 text-sm leading-relaxed">{explanation}</p>
      )}
      {!loadingExplain && !explanation && (
        <p className="text-ink-400 text-xs italic">Explanation unavailable.</p>
      )}

      <div className="flex gap-2 pt-1">
        <input
          type="text"
          className="flex-1 border border-ink-400/30 rounded-lg px-3 py-1.5 text-xs text-ink-900 focus:outline-none focus:ring-2 focus:ring-brand-500"
          placeholder="Ask about this match…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
          disabled={asking}
        />
        <button
          onClick={handleAsk}
          disabled={asking || !question.trim()}
          className="px-3 py-1.5 bg-ink-900 text-white text-xs font-medium rounded-lg hover:bg-ink-800 disabled:opacity-40 transition-colors"
        >
          Ask
        </button>
      </div>

      {asking && <p className="text-ink-400 text-xs italic">Thinking…</p>}
      {answer && (
        <p className="text-ink-800 text-sm leading-relaxed bg-ink-900/[0.03] rounded-lg px-3 py-2">
          {answer}
        </p>
      )}
    </div>
  );
}
