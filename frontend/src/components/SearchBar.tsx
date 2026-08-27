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
  onFileSubmit: (file: File) => void;
  onTextSubmit: (q: string) => void;
}

export default function SearchBar({ stage, onFileSubmit, onTextSubmit }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [textQuery, setTextQuery] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const busy = stage !== "idle" && stage !== "done" && stage !== "error";

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) onFileSubmit(file);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onFileSubmit(file);
  }

  function handleTextKey(e: React.KeyboardEvent) {
    if (e.key === "Enter" && textQuery.trim()) {
      onTextSubmit(textQuery.trim());
    }
  }

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 hover:border-gray-400 bg-white"
        } ${busy ? "opacity-60 pointer-events-none" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".step,.stp,.STEP"
          className="hidden"
          onChange={handleFileChange}
        />
        <div className="text-3xl mb-2">📐</div>
        <p className="text-sm font-medium text-gray-700">
          Drop a STEP file here or click to browse
        </p>
        <p className="text-xs text-gray-400 mt-1">.step / .stp · max 25 MB</p>
      </div>

      {/* Stage spinner */}
      {busy && (
        <div className="flex items-center gap-2 text-sm text-blue-600">
          <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
          </svg>
          {STAGE_LABELS[stage]}
        </div>
      )}

      {/* Divider */}
      <div className="flex items-center gap-2 text-xs text-gray-400">
        <div className="flex-1 border-t border-gray-200" />
        or search by text
        <div className="flex-1 border-t border-gray-200" />
      </div>

      {/* Text input */}
      <div className="flex gap-2">
        <input
          type="text"
          className="flex-1 border border-gray-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="e.g. M6 aluminum counterbore bracket"
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          onKeyDown={handleTextKey}
          disabled={busy}
        />
        <button
          className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:opacity-50"
          onClick={() => textQuery.trim() && onTextSubmit(textQuery.trim())}
          disabled={busy || !textQuery.trim()}
        >
          Search
        </button>
      </div>
    </div>
  );
}
