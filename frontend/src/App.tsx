import { useEffect, useState } from "react";
import type { Exchange } from "./types";
import { ask } from "./api";

const STARTER_PROMPTS = [
  "For how many days is it going to rain in Chennai?",
  "What was the weather in London a year ago?",
  "Which Asian cities will be rainiest this week?",
  "Compare Delhi and Bangalore right now",
  "Is it the best time to visit Bangkok now?",
  "How much did it rain in Pune last Christmas?", // Pune isn't tracked; tools geocode it
];

const STORAGE_KEY = "weather-rag-history";

function loadHistory(): Exchange[] {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
  } catch {
    return [];
  }
}

export default function App() {
  const [history, setHistory] = useState<Exchange[]>(loadHistory);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history.slice(-30)));
    } catch {
      /* storage full or unavailable: history just won't survive a refresh */
    }
  }, [history]);

  async function handleAsk(question: string) {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setInput("");
    setLoading(true);
    setError(null);

    // Append the exchange now, then fill it in as events stream in.
    setHistory((prev) => [...prev, { question: trimmed, answer: "", retrieved: [], tools: [] }]);
    const update = (fn: (ex: Exchange) => Exchange) =>
      setHistory((prev) => prev.map((ex, i) => (i === prev.length - 1 ? fn(ex) : ex)));

    try {
      await ask(trimmed, (event) => {
        if (event.type === "retrieved") update((ex) => ({ ...ex, retrieved: event.retrieved }));
        else if (event.type === "tool") update((ex) => ({ ...ex, tools: [...ex.tools, { name: event.name, arguments: event.arguments }] }));
        else if (event.type === "token") update((ex) => ({ ...ex, answer: ex.answer + event.text }));
        else if (event.type === "error") setError(event.error);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Weather RAG</h1>
        <p>
          English question in → embed → retrieve the closest live weather notes → the LLM reasons
          over them (and calls tools for rankings, any date, and travel advice) → answer out.
          ~100 cities across six regions, refreshed hourly from Open-Meteo.
        </p>
      </header>

      <div className="log">
        {history.map((ex, i) => (
          <div key={i} className="exchange">
            <div className="question">{ex.question}</div>
            {ex.tools.length > 0 && (
              <div className="tools">
                {ex.tools.map((t, j) => (
                  <code key={j} title={t.arguments}>
                    tool: {t.name}({t.arguments})
                  </code>
                ))}
              </div>
            )}
            {(ex.answer || !loading) && <div className="answer">{ex.answer}</div>}
            <div className="retrieved">
              {ex.retrieved.map((r, j) => (
                <details key={j} className="retrieved-item">
                  <summary>
                    retrieved <code>{r.city}</code> · {r.kind.replace("_", " ")} (similarity {r.score.toFixed(3)})
                  </summary>
                  <div className="retrieved-text">{r.text}</div>
                </details>
              ))}
            </div>
          </div>
        ))}
        {loading && !history[history.length - 1]?.answer && <div className="answer typing">thinking…</div>}
        {error && <div className="error">{error}</div>}
      </div>

      <div className="starters">
        {STARTER_PROMPTS.map((p) => (
          <button key={p} onClick={() => handleAsk(p)} disabled={loading}>
            {p}
          </button>
        ))}
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          handleAsk(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the weather somewhere…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
