import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  AlertOctagon,
  CheckCircle2,
  Info,
  Loader2,
  PlugZap,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { PageHeader, DemoBanner } from "@/components/PageHeader";
import { analyzeTransaction, isApiConfigured, type PredictionResult } from "@/lib/fraud-api";
import { apiKeys } from "@/lib/api-queries";
import { formatAmount, formatTimestamp } from "@/lib/demo-data";

export const Route = createFileRoute("/analyze")({
  head: () => ({
    meta: [
      { title: "Analyze Transaction — FraudLens AI Detection" },
      {
        name: "description",
        content:
          "Submit live transaction details to the Scikit-learn Random Forest model via Flask for real-time fraud scoring.",
      },
      { property: "og:title", content: "Analyze Transaction — FraudLens AI Detection" },
      {
        property: "og:description",
        content: "Submit transaction details for real-time AI fraud detection and risk assessment.",
      },
    ],
  }),
  component: AnalyzePage,
});

type FormState = {
  transactionId: string;
  amount: string;
  channel: string;
  merchantCategory: string;
  country: string;
};

const initialForm: FormState = {
  transactionId: "",
  amount: "",
  channel: "UPI",
  merchantCategory: "",
  country: "India",
};

const channels = ["UPI", "Card", "Transfer", "Wallet", "ATM"];

const merchantCategories = [
  "Groceries",
  "Restaurants",
  "Electronics",
  "Clothing",
  "Travel",
  "Fuel",
  "Healthcare",
  "Entertainment",
  "Utilities",
  "Other",
];

const countries = [
  "India",
  "United States",
  "United Kingdom",
  "Germany",
  "France",
  "Canada",
  "Australia",
  "Singapore",
  "United Arab Emirates",
  "Brazil",
  "Japan",
  "Other",
];

function validate(form: FormState) {
  const errors: Partial<Record<keyof FormState, string>> = {};
  if (!form.transactionId.trim()) {
    errors.transactionId = "Transaction ID is required.";
  }
  const amount = Number(form.amount);
  if (!form.amount.trim()) {
    errors.amount = "Amount is required.";
  } else if (Number.isNaN(amount) || amount <= 0) {
    errors.amount = "Enter an amount greater than 0.";
  } else if (amount > 100_000_000) {
    errors.amount = "Amount exceeds 100,000,000 limit.";
  }

  if (!form.merchantCategory) {
    errors.merchantCategory = "Merchant category is required.";
  }
  if (!form.country) {
    errors.country = "Country is required.";
  }
  return errors;
}

function AnalyzePage() {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<FormState>(initialForm);
  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const set = (key: keyof FormState) => (value: string) =>
    setForm((prev) => ({ ...prev, [key]: value }));

  const generateSampleId = () => {
    const id = `TXN-${Math.floor(100000 + Math.random() * 900000)}`;
    set("transactionId")(id);
  };

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    const nextErrors = validate(form);
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length > 0) {
      setResult(null);
      return;
    }

    setLoading(true);
    setResult(null);

    const response = await analyzeTransaction({
      transactionId: form.transactionId.trim(),
      amount: Number(form.amount),
      channel: form.channel,
      merchantCategory: form.merchantCategory,
      country: form.country,
    });

    setResult(response);
    setLoading(false);

    if (response.status === "ok") {
      toast.success(`Transaction evaluated as ${response.prediction.label}`);
      // Refresh Dashboard, History, and Analytics data
      queryClient.invalidateQueries({ queryKey: apiKeys.all });
    } else if (response.status === "error") {
      toast.error(response.message);
    } else if (response.status === "not_connected") {
      toast.info(response.message);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
      <PageHeader
        title="Analyze Transaction"
        description="Submit transaction details to the trained Scikit-learn Random Forest model for real-time fraud assessment."
      />

      {isApiConfigured() ? (
        <div className="flex items-center gap-2 rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-800 dark:text-emerald-200">
          <ShieldCheck className="size-4 shrink-0 text-emerald-600" aria-hidden />
          <span>
            Connected to Flask Machine Learning API (port 5000). Real-time inference and SQLite
            audit logging active.
          </span>
        </div>
      ) : (
        <DemoBanner text="ML backend not configured. Set VITE_FLASK_API_URL in .env.local to enable live model inference." />
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <Card className="shadow-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
            <CardTitle className="text-base">Transaction Details</CardTitle>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={generateSampleId}
              className="h-8 text-xs text-muted-foreground"
            >
              Generate ID
            </Button>
          </CardHeader>
          <CardContent>
            <form onSubmit={onSubmit} noValidate className="grid gap-4 sm:grid-cols-2">
              <Field
                id="transactionId"
                label="Transaction ID"
                value={form.transactionId}
                onChange={set("transactionId")}
                error={errors.transactionId}
                placeholder="TXN-100245"
              />
              <Field
                id="amount"
                label="Amount (INR)"
                type="number"
                value={form.amount}
                onChange={set("amount")}
                error={errors.amount}
                placeholder="2500"
              />
              <div className="grid gap-2">
                <Label htmlFor="channel">Channel</Label>
                <select
                  id="channel"
                  value={form.channel}
                  onChange={(e) => set("channel")(e.target.value)}
                  className="h-10 w-full rounded-md border border-input bg-card px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  {channels.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="merchantCategory">Merchant Category</Label>
                <select
                  id="merchantCategory"
                  value={form.merchantCategory}
                  onChange={(e) => set("merchantCategory")(e.target.value)}
                  aria-invalid={Boolean(errors.merchantCategory)}
                  aria-describedby={errors.merchantCategory ? "merchantCategory-error" : undefined}
                  className="h-10 w-full rounded-md border border-input bg-card px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <option value="">Select a category</option>
                  {merchantCategories.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                {errors.merchantCategory ? (
                  <p id="merchantCategory-error" className="text-xs text-destructive">
                    {errors.merchantCategory}
                  </p>
                ) : null}
              </div>
              <div className="grid gap-2">
                <Label htmlFor="country">Country</Label>
                <select
                  id="country"
                  value={form.country}
                  onChange={(e) => set("country")(e.target.value)}
                  aria-invalid={Boolean(errors.country)}
                  aria-describedby={errors.country ? "country-error" : undefined}
                  className="h-10 w-full rounded-md border border-input bg-card px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <option value="">Select a country</option>
                  {countries.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
                {errors.country ? (
                  <p id="country-error" className="text-xs text-destructive">
                    {errors.country}
                  </p>
                ) : null}
              </div>

              <div className="flex flex-wrap gap-3 sm:col-span-2 pt-2">
                <Button type="submit" disabled={loading} className="min-w-28">
                  {loading ? (
                    <>
                      <Loader2 className="size-4 animate-spin mr-2" aria-hidden /> Analyzing…
                    </>
                  ) : (
                    "Analyze"
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    setForm(initialForm);
                    setErrors({});
                    setResult(null);
                  }}
                  disabled={loading}
                >
                  <RotateCcw className="size-4 mr-1" /> Reset
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        <Card className="shadow-card">
          <CardHeader>
            <CardTitle className="text-base">Prediction Result</CardTitle>
          </CardHeader>
          <CardContent aria-live="polite" className="text-sm">
            {loading ? (
              <div className="flex flex-col items-center justify-center gap-3 py-10 text-muted-foreground">
                <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
                <p>Sending transaction to Flask ML service…</p>
              </div>
            ) : result?.status === "not_connected" ? (
              <div className="rounded-xl border border-warning-border bg-warning-surface p-4">
                <p className="flex items-center gap-2 font-semibold text-warning-foreground">
                  <PlugZap className="size-4" aria-hidden /> {result.message}
                </p>
                <p className="mt-2 text-xs text-warning-foreground/90">
                  Please verify that the Flask backend is running on{" "}
                  <code className="font-mono">http://127.0.0.1:5000</code>.
                </p>
              </div>
            ) : result?.status === "ok" ? (
              <div className="flex flex-col gap-4">
                <div
                  className={`flex items-center justify-between rounded-xl border p-4 ${
                    result.prediction.label === "Fraudulent"
                      ? "border-destructive/30 bg-destructive/10"
                      : "border-emerald-500/30 bg-emerald-500/10"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {result.prediction.label === "Fraudulent" ? (
                      <AlertOctagon className="size-7 text-destructive" />
                    ) : (
                      <CheckCircle2 className="size-7 text-emerald-600" />
                    )}
                    <div>
                      <p className="text-xs uppercase tracking-wider text-muted-foreground">
                        Predicted Class
                      </p>
                      <p className="text-lg font-bold">{result.prediction.label}</p>
                    </div>
                  </div>
                  <Badge
                    variant={
                      result.prediction.riskLevel === "High"
                        ? "destructive"
                        : result.prediction.riskLevel === "Medium"
                          ? "outline"
                          : "default"
                    }
                    className="text-xs uppercase"
                  >
                    {result.prediction.riskLevel} Risk
                  </Badge>
                </div>

                <dl className="grid grid-cols-2 gap-3 rounded-xl border border-border p-4">
                  <dt className="text-muted-foreground">Fraud Probability</dt>
                  <dd className="font-semibold">
                    {result.prediction.fraudProbability != null
                      ? `${(result.prediction.fraudProbability * 100).toFixed(1)}%`
                      : "Unavailable"}
                  </dd>
                  <dt className="text-muted-foreground">Confidence</dt>
                  <dd className="font-semibold">
                    {result.prediction.confidence != null
                      ? `${(result.prediction.confidence * 100).toFixed(1)}%`
                      : "Unavailable"}
                  </dd>
                  <dt className="text-muted-foreground">Amount</dt>
                  <dd>{formatAmount(result.prediction.amount)}</dd>
                  <dt className="text-muted-foreground">Transaction ID</dt>
                  <dd
                    className="font-mono text-xs truncate"
                    title={result.prediction.transactionId}
                  >
                    {result.prediction.transactionId}
                  </dd>
                  <dt className="text-muted-foreground">
                    {result.prediction.hasTransactionTime
                      ? "Transaction Time"
                      : "Recorded / Analysis Time"}
                  </dt>
                  <dd className="text-xs">
                    <span className="font-medium text-foreground">
                      {formatTimestamp(result.prediction.timestamp)}
                    </span>
                    {!result.prediction.hasTransactionTime && (
                      <span className="text-[10px] text-muted-foreground block mt-0.5">
                        Payment time not specified; recorded at analysis
                      </span>
                    )}
                  </dd>
                  {result.prediction.hasTransactionTime && result.prediction.createdAt ? (
                    <>
                      <dt className="text-muted-foreground">Analyzed At</dt>
                      <dd className="text-xs text-muted-foreground">
                        {formatTimestamp(result.prediction.createdAt)}
                      </dd>
                    </>
                  ) : null}
                </dl>
              </div>
            ) : result?.status === "error" ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-4 text-destructive">
                <p className="font-semibold">Prediction Failed</p>
                <p className="mt-1 text-sm">{result.message}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 py-10 text-center text-muted-foreground">
                <Info className="size-8 opacity-40" />
                <p>Fill in the transaction details and select Analyze to evaluate.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function Field({
  id,
  label,
  value,
  onChange,
  error,
  placeholder,
  type = "text",
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  error?: string | undefined;
  placeholder?: string | undefined;
  type?: string | undefined;
}) {
  return (
    <div className="grid gap-2">
      <Label htmlFor={id}>{label}</Label>
      <Input
        id={id}
        type={type}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={Boolean(error)}
        aria-describedby={error ? `${id}-error` : undefined}
        className="bg-card"
      />
      {error ? (
        <p id={`${id}-error`} className="text-xs text-destructive">
          {error}
        </p>
      ) : null}
    </div>
  );
}
