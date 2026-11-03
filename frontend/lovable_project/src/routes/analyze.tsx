import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  AlertOctagon,
  CheckCircle2,
  Clock,
  Info,
  Loader2,
  PlugZap,
  RotateCcw,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
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
import { formatAmount, formatTimestamp, formatTransactionAmount } from "@/lib/demo-data";

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
  transactionType: string;
  merchantCategory: string;
  country: string;
};

const initialForm: FormState = {
  transactionId: "",
  amount: "",
  channel: "UPI",
  transactionType: "P2M",
  merchantCategory: "",
  country: "India",
};

const transactionTypes = [
  { value: "P2M", label: "P2M (Person-to-Merchant)" },
  { value: "P2P", label: "P2P (Peer-to-Peer)" },
];

const channels = ["UPI", "Card", "Transfer", "Wallet", "ATM"];

const merchantCategories = [
  "Kirana & Groceries",
  "Quick Commerce",
  "Food & Dining",
  "Shopping & Retail",
  "Bill Payments & Utilities",
  "Travel & Ticketing",
  "Peer-to-Peer Transfer",
  "Education",
  "Entertainment",
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

  if (!form.transactionType) {
    errors.transactionType = "Transaction type is required.";
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
      currency: "INR",
      channel: form.channel,
      transactionType: form.transactionType,
      merchantCategory: form.merchantCategory,
      country: form.country,
    });

    setResult(response);
    setLoading(false);

    if (response.status === "ok") {
      const isFraud = response.prediction.label === "Fraudulent";
      const probPct =
        response.prediction.fraudProbability != null
          ? `${(response.prediction.fraudProbability * 100).toFixed(1)}%`
          : "N/A";
      const desc = `Risk Level: ${response.prediction.riskLevel} (${probPct} fraud probability)`;

      if (isFraud) {
        toast.warning("Evaluation complete: Flagged as Fraudulent", { description: desc });
      } else {
        toast.success("Evaluation complete: Classified as Legitimate", { description: desc });
      }
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
            Connected to Flask Machine Learning API. Real-time Indian UPI model inference and SQLite
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
            <form onSubmit={onSubmit} noValidate className="space-y-4">
              {/* Group 1: Model Inputs */}
              <div className="space-y-3">
                <div className="flex items-center justify-between pb-1">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-foreground">
                      Active Model Inputs
                    </h3>
                    <p className="text-[11px] text-muted-foreground">
                      Directly evaluated by the Random Forest classifier
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className="border-primary/40 bg-primary/10 text-[10px] text-primary"
                  >
                    ML Pipeline Features
                  </Badge>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div>
                    <Field
                      id="amount"
                      label="Amount (INR / ₹)"
                      type="number"
                      value={form.amount}
                      onChange={set("amount")}
                      error={errors.amount}
                      placeholder="450.00"
                    />
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      The Random Forest model is trained on the Indian UPI benchmark dataset and
                      expects amounts in Indian Rupees (₹).
                    </p>
                  </div>

                  <div className="grid gap-2">
                    <Label htmlFor="transactionType">Transaction Type</Label>
                    <select
                      id="transactionType"
                      value={form.transactionType}
                      onChange={(e) => set("transactionType")(e.target.value)}
                      aria-invalid={Boolean(errors.transactionType)}
                      aria-describedby={
                        errors.transactionType ? "transactionType-error" : undefined
                      }
                      className="h-10 w-full rounded-md border border-input bg-card px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      {transactionTypes.map((t) => (
                        <option key={t.value} value={t.value}>
                          {t.label}
                        </option>
                      ))}
                    </select>
                    {errors.transactionType ? (
                      <p id="transactionType-error" className="text-xs text-destructive">
                        {errors.transactionType}
                      </p>
                    ) : (
                      <p className="text-[11px] text-muted-foreground">
                        P2M for merchant payments, P2P for peer transfers.
                      </p>
                    )}
                  </div>

                  <div className="grid gap-2 sm:col-span-2">
                    <Label htmlFor="merchantCategory">Merchant Category</Label>
                    <select
                      id="merchantCategory"
                      value={form.merchantCategory}
                      onChange={(e) => set("merchantCategory")(e.target.value)}
                      aria-invalid={Boolean(errors.merchantCategory)}
                      aria-describedby={
                        errors.merchantCategory ? "merchantCategory-error" : undefined
                      }
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
                    ) : form.merchantCategory === "Education" ? (
                      <p className="text-[11px] text-muted-foreground">
                        School fees, college fees, coaching, online courses, books and stationery.
                      </p>
                    ) : null}
                  </div>

                  <div className="flex items-start gap-2.5 rounded-lg border border-primary/20 bg-primary/5 p-3 text-xs text-foreground/85 leading-relaxed sm:col-span-2">
                    <Clock className="size-4 shrink-0 mt-0.5 text-primary" aria-hidden />
                    <div>
                      <span className="font-semibold text-foreground">
                        Analysis Time & IST Features:
                      </span>{" "}
                      Temporal model inputs (hour of day, day of month, calendar month, day of week,
                      weekend indicator) are derived automatically from the server clock in India
                      Standard Time (IST, UTC+05:30) at the moment of evaluation.
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-t border-border/70 my-2" />

              {/* Group 2: Contextual Audit Metadata */}
              <div className="space-y-3">
                <div className="flex items-center justify-between pb-1">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Contextual Audit Metadata
                    </h3>
                    <p className="text-[11px] text-muted-foreground">
                      Persisted for audit logging; not evaluated by the model
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className="border-muted text-[10px] text-muted-foreground"
                  >
                    Audit Log Only
                  </Badge>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <Field
                    id="transactionId"
                    label="Transaction ID"
                    value={form.transactionId}
                    onChange={set("transactionId")}
                    error={errors.transactionId}
                    placeholder="TXN-100245"
                  />
                  <div className="grid gap-2">
                    <Label htmlFor="channel">Payment Channel</Label>
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
                  <div className="grid gap-2 sm:col-span-2">
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

                  <div className="flex items-start gap-2.5 rounded-lg border border-border bg-muted/40 p-3 text-xs text-muted-foreground leading-relaxed sm:col-span-2">
                    <Info className="size-4 shrink-0 mt-0.5 text-muted-foreground" aria-hidden />
                    <div>
                      <span className="font-semibold text-foreground/90">
                        Payment Rail & Audit Notice:
                      </span>{" "}
                      The Indian UPI model directly evaluates amount (INR), transaction type
                      (P2M/P2P), merchant category, and IST temporal features. Payment channel (UPI,
                      Card, Wallet, Transfer, ATM) and country are preserved in the SQLite audit log
                      for compliance records.
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-3 pt-2">
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
                  Please verify that the Flask backend is running and reachable.
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
                  <dd>
                    {(() => {
                      const formatted = formatTransactionAmount(
                        result.prediction.amount,
                        result.prediction.currency || "INR",
                      );
                      return (
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold">{formatted.text}</span>
                          {formatted.isLegacy && (
                            <Badge variant="outline" className="text-[10px] border-dashed">
                              Legacy / Unspecified
                            </Badge>
                          )}
                        </div>
                      );
                    })()}
                  </dd>
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

                {result.prediction.explanations ? (
                  <div className="space-y-3 rounded-xl border border-border/70 bg-card p-4">
                    <div className="flex items-center justify-between pb-1 border-b border-border/50">
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-foreground">
                          Model Feature Attributions
                        </h4>
                        <p className="text-[11px] text-muted-foreground">
                          Decision-path probability deltas relative to baseline
                        </p>
                      </div>
                      <Badge
                        variant="outline"
                        className="text-[10px] border-primary/30 text-primary"
                      >
                        Threshold ≥ ±1.0%
                      </Badge>
                    </div>

                    {/* Baseline & Imputed Context Information */}
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 rounded-lg bg-muted/40 p-2.5 text-xs">
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted-foreground block">
                          Ensemble Prior Baseline
                        </span>
                        <span className="font-semibold text-foreground">
                          {(result.prediction.explanations.base_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted-foreground block">
                          Imputed Context Effect
                        </span>
                        <span className="font-semibold text-foreground">
                          {result.prediction.explanations.imputed_context_contribution > 0
                            ? "+"
                            : ""}
                          {(
                            result.prediction.explanations.imputed_context_contribution * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                      <div>
                        <span className="text-[10px] uppercase tracking-wider text-muted-foreground block">
                          Effective Baseline
                        </span>
                        <span className="font-semibold text-foreground">
                          {(
                            result.prediction.explanations.effective_baseline_probability * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                    </div>
                    <p className="text-[10px] text-muted-foreground leading-tight px-0.5">
                      Note: Feature attributions are computed from the ensemble decision tree paths
                      using the exact 8-feature Indian UPI schema.
                    </p>

                    {/* Factors List */}
                    <div className="space-y-2 pt-1">
                      <span className="text-[11px] font-medium text-foreground block">
                        Active Input Drivers (|Impact| ≥ 1.0%):
                      </span>
                      {result.prediction.explanations.factors &&
                      result.prediction.explanations.factors.length > 0 ? (
                        <div className="space-y-2">
                          {result.prediction.explanations.factors.map((factor) => {
                            const isIncrease = factor.direction === "increases_risk";
                            return (
                              <div
                                key={factor.feature}
                                className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 rounded-lg border border-border/60 bg-muted/20 p-2.5 text-xs transition-colors"
                              >
                                <div className="space-y-0.5">
                                  <div className="flex items-center gap-2">
                                    <span className="font-semibold text-foreground">
                                      {factor.feature}
                                    </span>
                                    <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
                                      Value: {String(factor.value)}
                                    </span>
                                  </div>
                                  <p className="text-[11px] text-muted-foreground leading-normal">
                                    {factor.description}
                                  </p>
                                </div>
                                <div className="flex items-center sm:flex-col sm:items-end gap-1.5 shrink-0">
                                  <span
                                    className={`inline-flex items-center gap-1 font-mono text-xs font-bold ${
                                      isIncrease
                                        ? "text-destructive"
                                        : "text-emerald-600 dark:text-emerald-400"
                                    }`}
                                  >
                                    {isIncrease ? (
                                      <TrendingUp className="size-3.5" aria-hidden />
                                    ) : (
                                      <TrendingDown className="size-3.5" aria-hidden />
                                    )}
                                    {factor.impact_percentage}
                                  </span>
                                  <span
                                    className={`text-[10px] font-medium uppercase tracking-wider ${
                                      isIncrease
                                        ? "text-destructive/90"
                                        : "text-emerald-600/90 dark:text-emerald-400/90"
                                    }`}
                                  >
                                    {isIncrease ? "Increases Risk" : "Decreases Risk"}
                                  </span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="rounded-lg border border-dashed border-border/80 p-3 text-center text-xs text-muted-foreground italic">
                          No individual active input shifted model fraud probability by more than
                          the ±1.0% threshold relative to baseline.
                        </p>
                      )}
                    </div>

                    {/* Non-causal Disclaimer */}
                    <div className="border-t border-border/50 pt-2 text-[10px] text-muted-foreground leading-relaxed italic">
                      {result.prediction.explanations.disclaimer}
                    </div>
                  </div>
                ) : null}

                <div className="rounded-lg border border-border/60 bg-muted/30 p-2.5 text-[11px] text-muted-foreground leading-relaxed">
                  <span className="font-semibold text-foreground/85">Evaluation basis:</span>{" "}
                  Classification is computed from amount (INR), transaction type, merchant category,
                  and IST temporal features. Channel ({result.prediction.channel || "UPI"}) and
                  Country ({result.prediction.country || "India"}) are recorded as contextual audit
                  metadata.
                </div>
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
