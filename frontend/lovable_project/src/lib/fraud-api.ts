// FraudLens Real API Client for Python Flask + Scikit-learn backend.
// - Base URL is derived from VITE_FLASK_API_URL.
// - Validates and normalizes responses so components receive clean, typed data.
// - Rejects NaN, Infinity, negative probabilities, and out-of-range metrics.
// - Never fabricates fake predictions on network failure.

// ---------- Configuration ----------

function readBaseUrl(): string {
  const envUrl =
    (typeof import.meta !== "undefined" && import.meta.env
      ? (import.meta.env["VITE_FLASK_API_URL"] as string | undefined) ||
        (import.meta.env.VITE_FLASK_API_URL as string | undefined)
      : undefined) || "http://127.0.0.1:5000";

  return envUrl.trim().replace(/\/+$/, "") || "http://127.0.0.1:5000";
}

export const FLASK_API_BASE_URL: string = readBaseUrl();

export function isApiConfigured(): boolean {
  return Boolean(FLASK_API_BASE_URL);
}

const REQUEST_TIMEOUT_MS = 15000;

// ---------- Types: Request ----------

export type TransactionInput = {
  transactionId: string;
  amount: number;
  channel: string;
  merchantCategory: string;
  country: string;
  transactionTime?: string;
  hoursSincePrevTxn?: number;
};

// ---------- Types: Responses (Standard Contract) ----------

export type RiskLevel = "Low" | "Medium" | "High";

export type ExplanationFactor = {
  feature: string;
  value: string | number;
  contribution: number;
  impact_percentage: string;
  direction: "increases_risk" | "decreases_risk";
  description: string;
};

export type PredictionExplanations = {
  attribution_method: string;
  target_class: string;
  base_probability: number;
  predicted_probability: number;
  imputed_context_contribution: number;
  effective_baseline_probability: number;
  threshold: number;
  factors: ExplanationFactor[];
  imputed_features_isolated: string[];
  reconstruction_error?: number;
  disclaimer: string;
};

/** POST /api/predict response */
export type Prediction = {
  transactionId: string;
  prediction: string;
  label: string;
  fraudProbability: number; // 0..1
  confidence: number; // 0..1
  riskLevel: RiskLevel;
  amount: number;
  timestamp: string; // ISO 8601
  transactionTime?: string | null;
  hasTransactionTime?: boolean;
  createdAt?: string;
  channel?: string;
  merchantCategory?: string;
  country?: string;
  hoursSincePrevTxn?: number;
  isFraud?: number;
  explanations?: PredictionExplanations | null;
};

/** GET /api/transactions item */
export type ApiTransaction = {
  transactionId: string;
  id: string;
  amount: number;
  timestamp: string;
  transactionTime?: string | null;
  hasTransactionTime?: boolean;
  createdAt?: string;
  channel: string;
  merchantCategory: string;
  country: string;
  hoursSincePrevTxn: number;
  label: string;
  prediction: string;
  confidence: number | null; // 0..1
  fraudProbability: number | null; // 0..1
  riskLevel: RiskLevel;
  isFraud: number;
};

/** GET /api/transactions list response */
export type TransactionListResponse = {
  transactions: ApiTransaction[];
  total: number;
};

/** GET /api/stats response */
export type Stats = {
  totalTransactions: number;
  legitimateTransactions: number;
  fraudulentTransactions: number;
  fraudRate: number; // percentage 0..100
};

/** GET /api/analytics response */
export type Analytics = {
  channelBreakdown: { channel: string; legitimate: number; flagged: number }[];
  labelDistribution: { name: string; value: number }[];
  transactionTrend: { day: string; transactions: number; flagged: number }[];
  amountBands: { band: string; count: number }[];
  totalTransactions?: number;
  fraudulentTransactions?: number;
  legitimateTransactions?: number;
};

/** GET /api/health response */
export type HealthStatus = {
  status: string;
  model_loaded: boolean;
  database?: string;
  model_name?: string;
  total_records?: number;
};

// ---------- Error Handling ----------

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
  missing_url: "ML backend API not configured (VITE_FLASK_API_URL missing)",
  unavailable:
    "The prediction service is currently unavailable. Ensure Flask is running on port 5000.",
  network:
    "Could not reach the prediction service. Check your connection to http://127.0.0.1:5000.",
  bad_request: "The service rejected the submitted transaction details. Please check the values.",
  not_found: "The requested API endpoint was not found on the server.",
  server: "The prediction service encountered an error processing the request.",
  invalid_response: "The service returned an unexpected response structure.",
  unknown: "An unexpected error occurred while communicating with the service.",
};

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | undefined;

  constructor(kind: ApiErrorKind, status?: number, customMessage?: string) {
    super(customMessage || FRIENDLY[kind]);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export function toFriendlyMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return FRIENDLY.unknown;
}

function kindForStatus(status: number): ApiErrorKind {
  if (status === 400 || status === 422) return "bad_request";
  if (status === 404) return "not_found";
  if (status === 502 || status === 503 || status === 504) return "unavailable";
  if (status >= 500) return "server";
  return "unknown";
}

// ---------- Core Fetch Helper ----------

async function requestJson(path: string, init?: RequestInit): Promise<unknown> {
  if (!FLASK_API_BASE_URL) throw new ApiError("missing_url");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response: Response;

  const headers: Record<string, string> = {
    Accept: "application/json",
  };
  // Only add Content-Type for requests with a payload (e.g., POST/PUT).
  // Do NOT add Content-Type on GET requests to preserve standard simple CORS requests.
  if (init?.body) {
    headers["Content-Type"] = "application/json";
  }

  try {
    response = await fetch(`${FLASK_API_BASE_URL}${path}`, {
      ...init,
      headers: {
        ...headers,
        ...init?.headers,
      },
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("unavailable");
    }
    throw new ApiError("network");
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    let serverMessage: string | undefined;
    try {
      const errJson = (await response.json()) as Record<string, unknown>;
      if (errJson && typeof errJson["error"] === "string") {
        serverMessage = errJson["error"];
      }
    } catch {
      // not JSON or unreadable
    }
    throw new ApiError(kindForStatus(response.status), response.status, serverMessage);
  }

  try {
    return await response.json();
  } catch {
    throw new ApiError("invalid_response", response.status, "Server did not return valid JSON.");
  }
}

// ---------- Normalizers & Validators ----------

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null;
const isFiniteNum = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v) && !Number.isNaN(v);

function parseRiskLevel(val: unknown, fraudProb: number): RiskLevel {
  const str = String(val || "").toLowerCase();
  if (str === "high") return "High";
  if (str === "medium") return "Medium";
  if (str === "low") return "Low";
  return fraudProb >= 0.7 ? "High" : fraudProb >= 0.3 ? "Medium" : "Low";
}

function normalizeProbability(val: unknown): number | null {
  if (!isFiniteNum(val)) return null;
  // If returned as percentage 0..100, normalize to 0..1
  const prob = val > 1.0 ? val / 100 : val;
  if (prob < 0 || prob > 1 || !Number.isFinite(prob)) return null;
  return Math.max(0, Math.min(1, prob));
}

export function normalizePrediction(raw: unknown): Prediction {
  if (!isObj(raw)) {
    throw new ApiError("invalid_response", undefined, "Prediction response must be a JSON object.");
  }

  const transactionId = String(
    raw["transactionId"] || raw["transaction_id"] || raw["id"] || "",
  ).trim();
  if (!transactionId) {
    throw new ApiError(
      "invalid_response",
      undefined,
      "Missing transaction ID in prediction response.",
    );
  }

  const rawAmount = Number(raw["amount"] ?? raw["amt"]);
  if (!Number.isFinite(rawAmount) || rawAmount <= 0) {
    throw new ApiError("invalid_response", undefined, "Prediction returned an invalid amount.");
  }

  const label = String(raw["label"] || raw["prediction"] || "Legitimate").trim();
  const prediction = String(raw["prediction"] || label).trim();

  // Normalize fraud probability (0..1)
  const rawProb = raw["fraudProbability"] ?? raw["fraud_probability"];
  const fraudProbability = normalizeProbability(rawProb) ?? (label === "Fraudulent" ? 1.0 : 0.0);

  // Normalize confidence (0..1)
  const rawConf = raw["confidence"];
  let confidence = normalizeProbability(rawConf);
  if (confidence === null) {
    confidence = label === "Fraudulent" ? fraudProbability : 1.0 - fraudProbability;
  }
  confidence = Math.max(0, Math.min(1, confidence));

  const riskLevel = parseRiskLevel(raw["riskLevel"] ?? raw["risk_level"], fraudProbability);

  const rawTimestamp = String(
    raw["transactionTime"] || raw["timestamp"] || raw["created_at"] || "",
  ).trim();
  const validTimestamp =
    rawTimestamp && !Number.isNaN(Date.parse(rawTimestamp))
      ? rawTimestamp
      : new Date().toISOString();

  const rawCreatedAt = String(raw["createdAt"] || raw["created_at"] || "").trim();
  const createdAt =
    rawCreatedAt && !Number.isNaN(Date.parse(rawCreatedAt)) ? rawCreatedAt : undefined;

  const hasTransactionTime = Boolean(
    raw["hasTransactionTime"] ?? raw["has_transaction_time"] ?? false,
  );
  const transactionTime = hasTransactionTime ? validTimestamp : null;

  // Normalize optional explanation breakdown
  let explanations: PredictionExplanations | null = null;
  if (isObj(raw["explanations"])) {
    const rawExp = raw["explanations"] as Record<string, unknown>;
    const rawFactors = Array.isArray(rawExp["factors"]) ? rawExp["factors"] : [];
    const normalizedFactors: ExplanationFactor[] = [];

    for (const item of rawFactors) {
      if (isObj(item)) {
        const feature = String(item["feature"] || "").trim();
        const rawVal = item["value"];
        const value =
          typeof rawVal === "number" || typeof rawVal === "string" ? rawVal : String(rawVal ?? "");
        const contrib = Number(item["contribution"] ?? 0);
        const impact_percentage = String(
          item["impact_percentage"] || `${(contrib * 100).toFixed(1)}%`,
        );
        const direction: "increases_risk" | "decreases_risk" =
          item["direction"] === "increases_risk" || contrib > 0
            ? "increases_risk"
            : "decreases_risk";
        const description = String(item["description"] || "");

        if (feature) {
          normalizedFactors.push({
            feature,
            value,
            contribution: Number.isFinite(contrib) ? contrib : 0,
            impact_percentage,
            direction,
            description,
          });
        }
      }
    }

    explanations = {
      attribution_method: String(
        rawExp["attribution_method"] || "Decision Tree Path Decomposition",
      ),
      target_class: String(rawExp["target_class"] || "Fraudulent"),
      base_probability: Number(rawExp["base_probability"] ?? 0.5),
      predicted_probability: Number(rawExp["predicted_probability"] ?? fraudProbability),
      imputed_context_contribution: Number(rawExp["imputed_context_contribution"] ?? 0),
      effective_baseline_probability: Number(
        rawExp["effective_baseline_probability"] ??
          Number(rawExp["base_probability"] ?? 0.5) +
            Number(rawExp["imputed_context_contribution"] ?? 0),
      ),
      threshold: Number(rawExp["threshold"] ?? 0.01),
      factors: normalizedFactors,
      imputed_features_isolated: Array.isArray(rawExp["imputed_features_isolated"])
        ? rawExp["imputed_features_isolated"].map(String)
        : [],
      reconstruction_error:
        typeof rawExp["reconstruction_error"] === "number"
          ? rawExp["reconstruction_error"]
          : undefined,
      disclaimer: String(
        rawExp["disclaimer"] ||
          "Model attributions quantify how feature values shifted the prediction relative to the training prior and do not represent confirmed external causes.",
      ),
    };
  }

  return {
    transactionId,
    prediction,
    label,
    fraudProbability,
    confidence,
    riskLevel,
    amount: rawAmount,
    timestamp: validTimestamp,
    transactionTime,
    hasTransactionTime,
    createdAt,
    channel: typeof raw["channel"] === "string" ? raw["channel"] : "Card",
    merchantCategory:
      typeof raw["merchantCategory"] === "string"
        ? raw["merchantCategory"]
        : (raw["merchant_category"] as string) || "Other",
    country: typeof raw["country"] === "string" ? raw["country"] : "Other",
    hoursSincePrevTxn: isFiniteNum(raw["hoursSincePrevTxn"]) ? raw["hoursSincePrevTxn"] : 0,
    isFraud: raw["is_fraud"] === 1 || raw["isFraud"] === 1 || label === "Fraudulent" ? 1 : 0,
    explanations,
  };
}

export function normalizeTransactionItem(raw: unknown): ApiTransaction {
  if (!isObj(raw)) {
    throw new ApiError("invalid_response", undefined, "Transaction item is not a valid object.");
  }

  const transactionId = String(
    raw["transactionId"] || raw["transaction_id"] || raw["id"] || "",
  ).trim();
  const amount = Number(raw["amount"] ?? 0);
  const label = String(raw["label"] || raw["prediction"] || "Legitimate");
  const prediction = String(raw["prediction"] || label);
  const fraudProb = normalizeProbability(raw["fraudProbability"] ?? raw["fraud_probability"]);
  const confidence = normalizeProbability(raw["confidence"]);
  const riskLevel = parseRiskLevel(raw["riskLevel"] ?? raw["risk_level"], fraudProb ?? 0);
  const rawTime = String(raw["timestamp"] || raw["created_at"] || "").trim();
  const timestamp =
    rawTime && !Number.isNaN(Date.parse(rawTime)) ? rawTime : new Date().toISOString();

  const rawCreatedAt = String(raw["createdAt"] || raw["created_at"] || "").trim();
  const createdAt =
    rawCreatedAt && !Number.isNaN(Date.parse(rawCreatedAt)) ? rawCreatedAt : undefined;

  let hasTransactionTime = Boolean(
    raw["hasTransactionTime"] ?? raw["has_transaction_time"] ?? false,
  );
  if (!hasTransactionTime && timestamp && createdAt) {
    const tTime = Date.parse(timestamp);
    const cTime = Date.parse(createdAt);
    if (!Number.isNaN(tTime) && !Number.isNaN(cTime)) {
      if (Math.abs(tTime - cTime) > 60_000) {
        hasTransactionTime = true;
      }
    }
  }
  const transactionTime = hasTransactionTime ? timestamp : null;

  return {
    transactionId,
    id: transactionId,
    amount: Number.isFinite(amount) ? amount : 0,
    timestamp,
    transactionTime,
    hasTransactionTime,
    createdAt,
    channel: typeof raw["channel"] === "string" ? raw["channel"] : "Card",
    merchantCategory: String(raw["merchantCategory"] || raw["merchant_category"] || "Other"),
    country: String(raw["country"] || "Other"),
    hoursSincePrevTxn: Number(raw["hoursSincePrevTxn"] ?? raw["hours_since_prev_txn"] ?? 0),
    label,
    prediction,
    confidence,
    fraudProbability: fraudProb,
    riskLevel,
    isFraud: raw["is_fraud"] === 1 || raw["isFraud"] === 1 || label === "Fraudulent" ? 1 : 0,
  };
}

// ---------- API Endpoint Functions ----------

export async function predictTransaction(input: TransactionInput): Promise<Prediction> {
  const json = await requestJson("/api/predict", {
    method: "POST",
    body: JSON.stringify(input),
  });
  return normalizePrediction(json);
}

export async function getTransactions(limit = 100): Promise<TransactionListResponse> {
  const json = await requestJson(`/api/transactions?limit=${limit}`);

  if (Array.isArray(json)) {
    const list = json.map(normalizeTransactionItem);
    return { transactions: list, total: list.length };
  }

  if (isObj(json) && Array.isArray(json["transactions"])) {
    const list = json["transactions"].map(normalizeTransactionItem);
    const total = isFiniteNum(json["total"]) ? json["total"] : list.length;
    return { transactions: list, total };
  }

  return { transactions: [], total: 0 };
}

export async function getStats(): Promise<Stats> {
  const json = await requestJson("/api/stats");

  if (!isObj(json)) {
    throw new ApiError("invalid_response", undefined, "Stats response must be an object.");
  }

  const total = Number(json["totalTransactions"] ?? json["total_transactions"] ?? 0);
  const legit = Number(json["legitimateTransactions"] ?? json["legitimate_count"] ?? 0);
  const fraud = Number(json["fraudulentTransactions"] ?? json["fraud_count"] ?? 0);
  const rate = Number(json["fraudRate"] ?? json["fraud_rate"] ?? 0);

  return {
    totalTransactions: Math.max(0, isFiniteNum(total) ? total : 0),
    legitimateTransactions: Math.max(0, isFiniteNum(legit) ? legit : 0),
    fraudulentTransactions: Math.max(0, isFiniteNum(fraud) ? fraud : 0),
    fraudRate: isFiniteNum(rate) ? Math.max(0, Math.min(100, Math.round(rate * 100) / 100)) : 0,
  };
}

export async function getAnalytics(): Promise<Analytics> {
  const json = await requestJson("/api/analytics");

  if (!isObj(json)) {
    throw new ApiError("invalid_response", undefined, "Analytics response must be an object.");
  }

  const channelBreakdown = Array.isArray(json["channelBreakdown"])
    ? (json["channelBreakdown"] as { channel: string; legitimate: number; flagged: number }[])
    : [];

  const labelDistribution = Array.isArray(json["labelDistribution"])
    ? (json["labelDistribution"] as { name: string; value: number }[])
    : [];

  const transactionTrend = Array.isArray(json["transactionTrend"])
    ? (json["transactionTrend"] as { day: string; transactions: number; flagged: number }[])
    : [];

  const amountBands = Array.isArray(json["amountBands"])
    ? (json["amountBands"] as { band: string; count: number }[])
    : [];

  return {
    channelBreakdown,
    labelDistribution,
    transactionTrend,
    amountBands,
    totalTransactions: Number(json["totalTransactions"] ?? 0),
    fraudulentTransactions: Number(json["fraudulentTransactions"] ?? 0),
    legitimateTransactions: Number(json["legitimateTransactions"] ?? 0),
  };
}

export async function checkHealth(): Promise<HealthStatus> {
  const json = await requestJson("/api/health");
  if (!isObj(json))
    throw new ApiError("invalid_response", undefined, "Health response not an object.");

  return {
    status: String(json["status"] || "unknown"),
    model_loaded: Boolean(json["model_loaded"]),
    database: typeof json["database"] === "string" ? json["database"] : undefined,
    model_name: typeof json["model_name"] === "string" ? json["model_name"] : undefined,
    total_records: isFiniteNum(json["total_records"]) ? json["total_records"] : undefined,
  };
}

// ---------- Analyze Page Result Wrapper ----------

export type PredictionResult =
  | { status: "not_connected"; message: string }
  | { status: "ok"; prediction: Prediction }
  | { status: "error"; kind: ApiErrorKind; message: string };

export async function analyzeTransaction(input: TransactionInput): Promise<PredictionResult> {
  if (!isApiConfigured()) {
    return { status: "not_connected", message: FRIENDLY.missing_url };
  }
  try {
    const prediction = await predictTransaction(input);
    return { status: "ok", prediction };
  } catch (err) {
    const kind = err instanceof ApiError ? err.kind : "unknown";
    if (kind === "network" || kind === "unavailable" || kind === "missing_url") {
      return { status: "not_connected", message: toFriendlyMessage(err) };
    }
    return { status: "error", kind, message: toFriendlyMessage(err) };
  }
}
