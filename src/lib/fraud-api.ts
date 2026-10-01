// REAL API CLIENT for the future Python Flask + Scikit-learn backend.
// - Base URL comes only from VITE_FLASK_API_URL (no hardcoded hosts).
// - No fake responses: when the URL is missing or the backend is unreachable,
//   callers receive a typed error and the UI must say "not connected".
// - Demo data lives separately in ./demo-data.ts and is never returned from here.

// ---------- Configuration ----------

function readBaseUrl(): string | undefined {
  const raw = import.meta.env["VITE_FLASK_API_URL"] as string | undefined;
  const trimmed = raw?.trim();
  return trimmed ? trimmed.replace(/\/+$/, "") : undefined;
}

export const FLASK_API_BASE_URL: string | undefined = readBaseUrl();

export function isApiConfigured(): boolean {
  return Boolean(FLASK_API_BASE_URL);
}

const REQUEST_TIMEOUT_MS = 15000;

// ---------- Types: request ----------

// NOTE: fields mirror the Analyze form and the selected fraud dataset
// (transaction_id, amount, merchant_category, country, channel,
// hours_since_prev_txn). user_id and device_id are intentionally excluded
// until the model architecture handles them server-side.
export type TransactionInput = {
  transactionId: string;
  amount: number;
  channel: string;
  merchantCategory: string;
  country: string;
  hoursSincePrevTxn: number;
};

// ---------- Types: responses (endpoint contract) ----------

export type RiskLevel = "low" | "medium" | "high";

/** POST /api/predict */
export type Prediction = {
  transactionId: string;
  label: string;
  confidence: number; // 0..1
  timestamp: string; // ISO 8601
  amount: number;
  riskLevel?: RiskLevel;
};

/** GET /api/transactions */
export type ApiTransaction = {
  transactionId: string;
  amount: number;
  timestamp: string;
  channel: string;
  label: string;
  confidence: number | null;
  riskLevel: RiskLevel | null;
};

/** GET /api/stats */
export type Stats = {
  totalTransactions: number;
  legitimateTransactions: number;
  fraudulentTransactions: number;
  fraudRate: number; // percentage 0..100
};

/** GET /api/analytics */
export type Analytics = {
  channelBreakdown: { channel: string; legitimate: number; flagged: number }[];
  labelDistribution: { name: string; value: number }[];
  transactionTrend: { day: string; transactions: number; flagged: number }[];
  amountBands: { band: string; count: number }[];
};

// ---------- Errors ----------

export type ApiErrorKind =
  | "missing_url"
  | "unavailable"
  | "network"
  | "bad_request"
  | "not_found"
  | "server"
  | "invalid_response"
  | "unknown";

const FRIENDLY: Record<ApiErrorKind, string> = {
  missing_url: "ML model not connected yet",
  unavailable: "The prediction service is currently unavailable. Please try again later.",
  network: "Could not reach the prediction service. Check your connection and try again.",
  bad_request: "The service rejected the submitted details. Please review the form.",
  not_found: "The requested service endpoint was not found.",
  server: "The prediction service encountered an error. Please try again later.",
  invalid_response: "The service returned an unexpected response.",
  unknown: "Something went wrong. Please try again.",
};

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | undefined;
  constructor(kind: ApiErrorKind, status?: number) {
    super(FRIENDLY[kind]);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export function toFriendlyMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : FRIENDLY.unknown;
}

function kindForStatus(status: number): ApiErrorKind {
  if (status === 400 || status === 422) return "bad_request";
  if (status === 404) return "not_found";
  if (status === 502 || status === 503 || status === 504) return "unavailable";
  if (status >= 500) return "server";
  return "unknown";
}

// ---------- Core request helper ----------

async function request<T>(
  path: string,
  validate: (data: unknown) => data is T,
  init?: RequestInit,
): Promise<T> {
  if (!FLASK_API_BASE_URL) throw new ApiError("missing_url");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response: Response;
  try {
    response = await fetch(`${FLASK_API_BASE_URL}${path}`, {
      ...init,
      headers: { Accept: "application/json", "Content-Type": "application/json", ...init?.headers },
      signal: controller.signal,
    });
  } catch (err) {
    // Fetch rejects on DNS/CORS/refused connections and on timeout abort.
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("unavailable");
    }
    throw new ApiError("network");
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) throw new ApiError(kindForStatus(response.status), response.status);

  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("invalid_response", response.status);
  }
  if (!validate(data)) throw new ApiError("invalid_response", response.status);
  return data;
}

// ---------- Runtime validators (lightweight, no extra deps) ----------

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === "string";

function isPrediction(v: unknown): v is Prediction {
  return (
    isObj(v) &&
    isStr(v["transactionId"]) &&
    isStr(v["label"]) &&
    isNum(v["confidence"]) &&
    isStr(v["timestamp"]) &&
    isNum(v["amount"])
  );
}

function isTransactionList(v: unknown): v is ApiTransaction[] {
  return (
    Array.isArray(v) &&
    v.every(
      (t) =>
        isObj(t) &&
        isStr(t["transactionId"]) &&
        isNum(t["amount"]) &&
        isStr(t["timestamp"]) &&
        isStr(t["channel"]) &&
        isStr(t["label"]),
    )
  );
}

function isStats(v: unknown): v is Stats {
  return (
    isObj(v) &&
    isNum(v["totalTransactions"]) &&
    isNum(v["legitimateTransactions"]) &&
    isNum(v["fraudulentTransactions"]) &&
    isNum(v["fraudRate"])
  );
}

function isAnalytics(v: unknown): v is Analytics {
  return (
    isObj(v) &&
    Array.isArray(v["channelBreakdown"]) &&
    Array.isArray(v["labelDistribution"]) &&
    Array.isArray(v["transactionTrend"]) &&
    Array.isArray(v["amountBands"])
  );
}

// ---------- Endpoint functions ----------

export function predictTransaction(input: TransactionInput): Promise<Prediction> {
  return request("/api/predict", isPrediction, { method: "POST", body: JSON.stringify(input) });
}

export function getTransactions(): Promise<ApiTransaction[]> {
  return request("/api/transactions", isTransactionList);
}

export function getStats(): Promise<Stats> {
  return request("/api/stats", isStats);
}

export function getAnalytics(): Promise<Analytics> {
  return request("/api/analytics", isAnalytics);
}

// ---------- Analyze-page wrapper (no fake predictions) ----------

export type PredictionResult =
  | { status: "not_connected"; message: string }
  | { status: "ok"; prediction: Prediction }
  | { status: "error"; kind: ApiErrorKind; message: string };

export async function analyzeTransaction(input: TransactionInput): Promise<PredictionResult> {
  if (!isApiConfigured()) {
    return { status: "not_connected", message: FRIENDLY.missing_url };
  }
  try {
    return { status: "ok", prediction: await predictTransaction(input) };
  } catch (err) {
    const kind = err instanceof ApiError ? err.kind : "unknown";
    if (kind === "network" || kind === "unavailable") {
      return { status: "not_connected", message: FRIENDLY.missing_url };
    }
    return { status: "error", kind, message: toFriendlyMessage(err) };
  }
}
