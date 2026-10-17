export interface RetrievedDoc {
  city: string;
  kind: "forecast" | "past_week" | "year_ago";
  text: string;
  score: number;
}

export interface ToolCall {
  name: string;
  arguments: string;
}

// One newline-delimited JSON event from POST /api/ask.
export type StreamEvent =
  | { type: "retrieved"; retrieved: RetrievedDoc[] }
  | ({ type: "tool" } & ToolCall)
  | { type: "token"; text: string }
  | { type: "done" }
  | { type: "error"; error: string };

export interface Exchange {
  question: string;
  answer: string;
  retrieved: RetrievedDoc[];
  tools: ToolCall[];
}
