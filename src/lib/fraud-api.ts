// Integration seam for the future Python Flask REST API.
// Today it deliberately returns a "not connected" result: the UI must never
// invent risk scores or predictions while no model is wired up.

export type TransactionInput = {
  transactionId: string;
  amount: number;
  channel: string;
  merchantCategory: string;
  originCountry: string;
  accountAgeDays: number;
};

export type PredictionResult =
  | { status: "not_connected"; message: string }
  | { status: "ok"; label: string; confidence: number }
  | { status: "error"; message: string };

export const FLASK_API_BASE_URL: string | undefined = undefined;

export async function analyzeTransaction(_input: TransactionInput): Promise<PredictionResult> {
  await new Promise((resolve) => setTimeout(resolve, 900));

  if (!FLASK_API_BASE_URL) {
    return {
      status: "not_connected",
      message: "ML model not connected yet",
    };
  }

  // Future: POST _input to `${FLASK_API_BASE_URL}/predict` and return the response.
  return { status: "error", message: "Prediction endpoint not implemented." };
}
