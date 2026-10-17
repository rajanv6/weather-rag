import type { StreamEvent } from "./types";

const API_BASE = "http://localhost:8080";

/** Posts a question and calls onEvent for each streamed event as it arrives. */
export async function ask(question: string, onEvent: (event: StreamEvent) => void): Promise<void> {
  const res = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(import.meta.env.VITE_API_KEY ? { "X-API-Key": import.meta.env.VITE_API_KEY } : {}),
    },
    body: JSON.stringify({ question }),
  });

  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Request failed with status ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? ""; // last piece may be an incomplete line
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line) as StreamEvent);
    }
  }
}
