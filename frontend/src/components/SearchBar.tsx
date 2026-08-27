import { useRef, useState } from "react";
import type { SearchStage } from "@/api/client";

const STAGE_LABELS: Record<SearchStage, string> = {
  idle: "",
  reading: "Reading STEP file…",
  graph: "Building B-rep graph…",
  uvnet: "UV-Net embedding…",
  searching: "Searching index…",
  done: "",
  error: "",
};

interface Props {
  stage: SearchStage;
  file: File | null;
  text: string;
  onFileChange: (file: File | null) => void;
  onTextChange: (text: string) => void;
  onSubmit: () => void;
}

export default function SearchBar({
  stage,
  file,
  text,
  onFileChange,
  onTextChange,
  onSubmit,
}: Props) {
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const busy = stage !== "idle" && stage !== "done" && stage !== "error";
  const canSubmit = (!!file || text.trim().length > 0) && !busy;

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files[0];
    if (f) onFileChange(f);
  }

  function handleFileInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) onFileChange(f);
  }

  function handleKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && canSubmit) onSubmit();
  }

  return (
    <div className="space-y-3">
      {/* Drop zone / file chip */}
      {file ? (
        <div className="flex items-center gap-2 border border-brand-100 bg-brand-50 rounded-xl px-4 py-3">
          <span className="text-lg">📐</span>
          <span className="flex-1 min-w-0 truncate text-sm font-medium text-ink-800">
            {file.name}
          </span>
          <button
            onClick={() => onFileChange(null)}
            className="text-ink-400 hover:text-ink-800 text-sm shrink-0"
            disabled={busy}
          >
            ✕
          </button>
        </div>
      ) : (
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            dragOver
              ? "border-brand-500 bg-brand-50"
              : "border-ink-400/30 hover:border-ink-400/60 bg-white"
          } ${busy ? "opacity-60 pointer-events-none" : ""}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".step,.stp,.STEP"
            className="hidden"
            onChange={handleFileInputChange}
          />
          <div className="text-3xl mb-2">📐</div>
          <p className="text-sm font-medium text-ink-800">
            Drop a STEP file here or click to browse
          </p>
          <p className="text-xs text-ink-400 mt-1">.step / .stp · max 25 MB</p>
        </div>
      )}

      {/* Stage spinner */}
      {busy && (
        <div className="flex items-center gap-2 text-sm text-brand-600">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
          </svg>
          {STAGE_LABELS[stage]}
        </div>
      )}

      {/* Divider */}
      <div className="flex items-center gap-2 text-xs text-ink-400">
        <div className="flex-1 border-t border-ink-400/20" />
        {file ? "narrow with text (optional)" : "or search by text"}
        <div className="flex-1 border-t border-ink-400/20" />
      </div>

      {/* Text input + submit */}
      <div className="flex gap-2">
        <input
          type="text"
          className="flex-1 border border-ink-400/30 rounded-lg px-3 py-2 text-sm text-ink-900 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-brand-500"
          placeholder="e.g. M6 aluminum counterbore bracket"
          value={text}
          onChange={(e) => onTextChange(e.target.value)}
          onKeyDown={handleKey}
          disabled={busy}
        />
        <button
          className="px-4 py-2 bg-brand-500 text-white text-sm font-medium rounded-lg hover:bg-brand-600 disabled:opacity-40 transition-colors"
          onClick={onSubmit}
          disabled={!canSubmit}
        >
          Search
        </button>
      </div>
    </div>
  );
}
