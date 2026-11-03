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
  Sparkles,
  Cpu,
  Layers,
  ArrowRight,
} from "lucide-react";
import { toast } from "sonner";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { PageHeader, DemoBanner } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { RiskBadge } from "@/components/RiskBadge";
import { analyzeTransaction, isApiConfigured, type PredictionResult } from "@/lib/fraud-api";
import { apiKeys } from "@/lib/api-queries";
import { formatTimestamp, formatTransactionAmount } from "@/lib/demo-data";

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

export function AnalyzePage() {
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
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
      <PageHeader
        title="Transaction Analysis & Scoring"
        description="Evaluate simulated transactions against the calibrated Indian UPI 80-tree Random Forest classifier for instant fraud scoring."
      />

      {isApiConfigured() ? (
        <div className="flex items-center gap-2.5 rounded-lg border border-emerald-500/25 bg-emerald-500/5 px-4 py-3 text-sm text-emerald-300">
          <ShieldCheck className="size-4 shrink-0 text-emerald-400" aria-hidden />
          <span className="text-xs sm:text-sm">
            Live Flask Machine Learning API connected. Active inference with SQLite audit logging
            enabled (academic prototype).
          </span>
        </div>
      ) : (
        <DemoBanner text="ML backend not configured. Set VITE_FLASK_API_URL in .env.local to enable live model inference." />
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        {/* Left Column: Form Card */}
        <Card className="border-border/80 bg-card shadow-card">
          <CardHeader className="flex flex-row items-center justify-between pb-4">
            <div>
              <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
                <Cpu className="size-4 text-primary" />
                Transaction Parameters
              </CardTitle>
              <CardDescription className="text-xs text-muted-foreground mt-0.5">
                Input transaction characteristics for real-time model evaluation
              </CardDescription>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={generateSampleId}
              className="h-7 text-xs border-border bg-elevated/60 hover:bg-elevated text-slate-300"
            >
              Generate ID
            </Button>
          </CardHeader>
          <CardContent>
            <form onSubmit={onSubmit} noValidate className="space-y-5">
              {/* Group 1: Model Inputs */}
              <div className="space-y-4 rounded-lg border border-border/60 bg-elevated/20 p-4">
                <div className="flex items-center justify-between pb-1 border-b border-border/40">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-primary">
                      1. Active Model Features
                    </h3>
                    <p className="text-[11px] text-muted-foreground">
                      Direct inputs passed into the Random Forest decision tree ensemble
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className="border-primary/40 bg-primary/10 text-[10px] text-primary font-mono"
                  >
                    Evaluated Features
                  </Badge>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="grid gap-1.5">
                    <Label htmlFor="amount" className="text-xs font-medium text-slate-300">
                      Amount (INR / ₹) <span className="text-red-400">*</span>
                    </Label>
                    <div className="relative">
                      <span className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3 text-muted-foreground font-mono text-xs">
                        ₹
                      </span>
                      <Input
                        id="amount"
                        type="number"
                        value={form.amount}
                        onChange={(e) => set("amount")(e.target.value)}
                        placeholder="450.00"
                        className="pl-7 bg-elevated/50 border-border text-foreground font-mono text-sm placeholder:text-muted-foreground focus-visible:ring-primary"
                        aria-invalid={Boolean(errors.amount)}
                        aria-describedby={errors.amount ? "amount-error" : undefined}
                      />
                    </div>
                    {errors.amount ? (
                      <p id="amount-error" className="text-xs text-red-400">
                        {errors.amount}
                      </p>
                    ) : (
                      <p className="text-[11px] text-muted-foreground">
                        Model expects amounts in Indian Rupees (₹).
                      </p>
                    )}
                  </div>

                  <div className="grid gap-1.5">
                    <Label htmlFor="transactionType" className="text-xs font-medium text-slate-300">
                      Transaction Type <span className="text-red-400">*</span>
                    </Label>
                    <select
                      id="transactionType"
                      value={form.transactionType}
                      onChange={(e) => set("transactionType")(e.target.value)}
                      aria-invalid={Boolean(errors.transactionType)}
                      aria-describedby={
                        errors.transactionType ? "transactionType-error" : undefined
                      }
                      className="h-9 w-full rounded-md border border-border bg-elevated/50 px-3 text-xs text-foreground outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                    >
                      {transactionTypes.map((t) => (
                        <option key={t.value} value={t.value} className="bg-card text-foreground">
                          {t.label}
                        </option>
                      ))}
                    </select>
                    {errors.transactionType ? (
                      <p id="transactionType-error" className="text-xs text-red-400">
                        {errors.transactionType}
                      </p>
                    ) : (
                      <p className="text-[11px] text-muted-foreground">
                        P2M for merchant purchases, P2P for peer transfers.
                      </p>
                    )}
                  </div>

                  <div className="grid gap-1.5 sm:col-span-2">
                    <Label
                      htmlFor="merchantCategory"
                      className="text-xs font-medium text-slate-300"
                    >
                      Merchant Category <span className="text-red-400">*</span>
                    </Label>
                    <select
                      id="merchantCategory"
                      value={form.merchantCategory}
                      onChange={(e) => set("merchantCategory")(e.target.value)}
                      aria-invalid={Boolean(errors.merchantCategory)}
                      aria-describedby={
                        errors.merchantCategory ? "merchantCategory-error" : undefined
                      }
                      className="h-9 w-full rounded-md border border-border bg-elevated/50 px-3 text-xs text-foreground outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                    >
                      <option value="" className="bg-card text-muted-foreground">
                        Select merchant category…
                      </option>
                      {merchantCategories.map((c) => (
                        <option key={c} value={c} className="bg-card text-foreground">
                          {c}
                        </option>
                      ))}
                    </select>
                    {errors.merchantCategory ? (
                      <p id="merchantCategory-error" className="text-xs text-red-400">
                        {errors.merchantCategory}
                      </p>
                    ) : form.merchantCategory === "Education" ? (
                      <p className="text-[11px] text-cyan-400">
                        Includes school fees, college tuition, coaching, online courses, and
                        academic supplies.
                      </p>
                    ) : null}
                  </div>

                  <div className="flex items-start gap-2.5 rounded-md border border-primary/20 bg-primary/5 p-2.5 text-xs text-slate-300 leading-relaxed sm:col-span-2">
                    <Clock className="size-4 shrink-0 mt-0.5 text-primary" aria-hidden />
                    <div>
                      <span className="font-semibold text-foreground">
                        Automated IST Temporal Inputs:
                      </span>{" "}
                      Hour of day, day of month, calendar month, day of week, and weekend indicator
                      are automatically extracted in India Standard Time (IST, UTC+05:30) upon
                      submission.
                    </div>
                  </div>
                </div>
              </div>

              {/* Group 2: Contextual Audit Metadata */}
              <div className="space-y-4 rounded-lg border border-border/60 bg-elevated/20 p-4">
                <div className="flex items-center justify-between pb-1 border-b border-border/40">
                  <div>
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      2. Contextual Audit Metadata
                    </h3>
                    <p className="text-[11px] text-muted-foreground">
                      Stored in the SQLite audit log for traceability and evaluation history
                    </p>
                  </div>
                  <Badge
                    variant="outline"
                    className="border-border text-[10px] text-muted-foreground font-mono"
                  >
                    Audit Trail
                  </Badge>
                </div>

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="grid gap-1.5">
                    <Label htmlFor="transactionId" className="text-xs font-medium text-slate-300">
                      Transaction ID <span className="text-red-400">*</span>
                    </Label>
                    <Input
                      id="transactionId"
                      value={form.transactionId}
                      onChange={(e) => set("transactionId")(e.target.value)}
                      placeholder="TXN-100245"
                      className="bg-elevated/50 border-border text-foreground font-mono text-xs placeholder:text-muted-foreground focus-visible:ring-primary"
                      aria-invalid={Boolean(errors.transactionId)}
                      aria-describedby={errors.transactionId ? "transactionId-error" : undefined}
                    />
                    {errors.transactionId ? (
                      <p id="transactionId-error" className="text-xs text-red-400">
                        {errors.transactionId}
                      </p>
                    ) : null}
                  </div>

                  <div className="grid gap-1.5">
                    <Label htmlFor="channel" className="text-xs font-medium text-slate-300">
                      Payment Channel
                    </Label>
                    <select
                      id="channel"
                      value={form.channel}
                      onChange={(e) => set("channel")(e.target.value)}
                      className="h-9 w-full rounded-md border border-border bg-elevated/50 px-3 text-xs text-foreground outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                    >
                      {channels.map((c) => (
                        <option key={c} value={c} className="bg-card text-foreground">
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="grid gap-1.5 sm:col-span-2">
                    <Label htmlFor="country" className="text-xs font-medium text-slate-300">
                      Country Origin <span className="text-red-400">*</span>
                    </Label>
                    <select
                      id="country"
                      value={form.country}
                      onChange={(e) => set("country")(e.target.value)}
                      aria-invalid={Boolean(errors.country)}
                      aria-describedby={errors.country ? "country-error" : undefined}
                      className="h-9 w-full rounded-md border border-border bg-elevated/50 px-3 text-xs text-foreground outline-none focus:border-primary focus:ring-1 focus:ring-primary"
                    >
                      <option value="" className="bg-card text-muted-foreground">
                        Select a country…
                      </option>
                      {countries.map((c) => (
                        <option key={c} value={c} className="bg-card text-foreground">
                          {c}
                        </option>
                      ))}
                    </select>
                    {errors.country ? (
                      <p id="country-error" className="text-xs text-red-400">
                        {errors.country}
                      </p>
                    ) : null}
                  </div>

                  <div className="flex items-start gap-2 rounded-md border border-border/40 bg-elevated/30 p-2.5 text-xs text-muted-foreground leading-relaxed sm:col-span-2">
                    <Info className="size-4 shrink-0 mt-0.5 text-muted-foreground" aria-hidden />
                    <div>
                      <span className="font-semibold text-slate-300">Multi-Rail Traceability:</span>{" "}
                      The model assesses amount, transaction type, merchant category, and temporal
                      signals. Payment channel (UPI, Card, Wallet, Transfer, ATM) and country are
                      preserved in the permanent SQLite audit repository.
                    </div>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <Button
                  type="submit"
                  disabled={loading}
                  className="min-w-32 bg-primary hover:bg-primary/90 text-white font-medium shadow-sm h-9 text-xs"
                >
                  {loading ? (
                    <>
                      <Loader2 className="size-3.5 animate-spin mr-2" aria-hidden /> Auditing…
                    </>
                  ) : (
                    <>
                      Run AI Assessment <ArrowRight className="size-3.5 ml-1.5" />
                    </>
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
                  className="h-9 text-xs border-border bg-card hover:bg-elevated text-slate-300"
                >
                  <RotateCcw className="size-3.5 mr-1.5" /> Reset Form
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Right Column: Prediction Result */}
        <Card className="border-border/80 bg-card shadow-card">
          <CardHeader className="pb-4">
            <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
              <Sparkles className="size-4 text-primary" />
              Machine Learning Verdict
            </CardTitle>
            <CardDescription className="text-xs text-muted-foreground mt-0.5">
              Instant inference score and explainable attribution factors
            </CardDescription>
          </CardHeader>
          <CardContent aria-live="polite" className="text-sm">
            {loading ? (
              <div className="flex flex-col items-center justify-center gap-3 py-16 text-muted-foreground">
                <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
                <p className="text-xs font-medium text-slate-300">
                  Evaluating transaction vectors against 80 decision trees…
                </p>
              </div>
            ) : result?.status === "not_connected" ? (
              <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-amber-300">
                <p className="flex items-center gap-2 font-semibold text-xs sm:text-sm">
                  <PlugZap className="size-4 text-amber-400" aria-hidden /> {result.message}
                </p>
                <p className="mt-1.5 text-xs text-amber-200/80">
                  Verify that the Flask server is running on{" "}
                  <code className="font-mono bg-black/30 px-1 py-0.5 rounded">
                    http://127.0.0.1:5000
                  </code>
                  .
                </p>
              </div>
            ) : result?.status === "ok" ? (
              <div className="flex flex-col gap-4">
                {/* Result Classification Banner */}
                <div
                  className={`flex items-center justify-between rounded-lg border p-4 ${
                    result.prediction.label === "Fraudulent"
                      ? "border-red-500/40 bg-red-950/20 shadow-[0_0_20px_rgba(239,68,68,0.1)]"
                      : "border-emerald-500/40 bg-emerald-950/20 shadow-[0_0_20px_rgba(16,185,129,0.1)]"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`grid size-10 place-items-center rounded-lg ${
                        result.prediction.label === "Fraudulent"
                          ? "bg-red-500/20 text-red-400 border border-red-500/30"
                          : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
                      }`}
                    >
                      {result.prediction.label === "Fraudulent" ? (
                        <AlertOctagon className="size-5" />
                      ) : (
                        <CheckCircle2 className="size-5" />
                      )}
                    </div>
                    <div>
                      <p className="text-[10px] uppercase font-semibold tracking-wider text-muted-foreground">
                        Predicted Classification
                      </p>
                      <p
                        className={`text-lg font-bold tracking-tight ${
                          result.prediction.label === "Fraudulent"
                            ? "text-red-400"
                            : "text-emerald-400"
                        }`}
                      >
                        {result.prediction.label}
                      </p>
                    </div>
                  </div>
                  <RiskBadge level={result.prediction.riskLevel} />
                </div>

                {/* Fraud Probability Progress Indicator */}
                {result.prediction.fraudProbability != null && (
                  <div className="space-y-1.5 rounded-lg border border-border/80 bg-elevated/40 p-3.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-300">Fraud Risk Probability</span>
                      <span className="font-mono font-bold text-foreground">
                        {(result.prediction.fraudProbability * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 w-full overflow-hidden rounded-full bg-background border border-border/60">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          result.prediction.fraudProbability > 0.5
                            ? "bg-red-500"
                            : result.prediction.fraudProbability > 0.2
                              ? "bg-amber-400"
                              : "bg-emerald-500"
                        }`}
                        style={{
                          width: `${Math.min(100, Math.max(2, result.prediction.fraudProbability * 100))}%`,
                        }}
                      />
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-0.5">
                      <span>Low Risk (0.00)</span>
                      <span>Decision Cutoff (0.50)</span>
                      <span>Critical (1.00)</span>
                    </div>
                  </div>
                )}

                {/* Key-Value Specification Grid */}
                <dl className="grid grid-cols-2 gap-3 rounded-lg border border-border/80 bg-elevated/30 p-3.5 text-xs">
                  <div>
                    <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      Confidence
                    </dt>
                    <dd className="font-semibold text-foreground font-mono mt-0.5">
                      {result.prediction.confidence != null
                        ? `${(result.prediction.confidence * 100).toFixed(1)}%`
                        : "Unavailable"}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      Amount
                    </dt>
                    <dd className="font-semibold text-foreground font-mono mt-0.5">
                      {(() => {
                        const formatted = formatTransactionAmount(
                          result.prediction.amount,
                          result.prediction.currency || "INR",
                        );
                        return (
                          <div className="flex items-center gap-1.5">
                            <span>{formatted.text}</span>
                            {formatted.isLegacy && (
                              <Badge
                                variant="outline"
                                className="text-[9px] border-border text-muted-foreground"
                              >
                                Legacy
                              </Badge>
                            )}
                          </div>
                        );
                      })()}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      Transaction ID
                    </dt>
                    <dd
                      className="font-mono text-xs text-slate-300 truncate mt-0.5"
                      title={result.prediction.transactionId}
                    >
                      {result.prediction.transactionId}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      {result.prediction.hasTransactionTime
                        ? "Transaction Time"
                        : "Audit Timestamp"}
                    </dt>
                    <dd className="text-xs text-foreground mt-0.5">
                      {formatTimestamp(result.prediction.timestamp)}
                    </dd>
                  </div>
                </dl>

                {/* Feature Attributions Panel */}
                {result.prediction.explanations && (
                  <div className="space-y-3 rounded-lg border border-border/80 bg-elevated/30 p-3.5">
                    <div className="flex items-center justify-between pb-1.5 border-b border-border/60">
                      <div>
                        <h4 className="text-xs font-semibold uppercase tracking-wider text-foreground flex items-center gap-1.5">
                          <Layers className="size-3.5 text-primary" />
                          Feature Attributions
                        </h4>
                        <p className="text-[11px] text-muted-foreground">
                          Decision-path probability deltas relative to baseline
                        </p>
                      </div>
                      <Badge
                        variant="outline"
                        className="text-[10px] border-border bg-background text-muted-foreground font-mono"
                      >
                        Cutoff ≥ ±1.0%
                      </Badge>
                    </div>

                    {/* Baseline mini-metrics */}
                    <div className="grid grid-cols-3 gap-2 rounded-md bg-background/80 border border-border/60 p-2.5 text-xs text-center">
                      <div>
                        <span className="text-[9px] uppercase tracking-wider text-muted-foreground block">
                          Prior Base
                        </span>
                        <span className="font-mono font-semibold text-foreground">
                          {(result.prediction.explanations.base_probability * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div>
                        <span className="text-[9px] uppercase tracking-wider text-muted-foreground block">
                          Imputed Shift
                        </span>
                        <span className="font-mono font-semibold text-foreground">
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
                        <span className="text-[9px] uppercase tracking-wider text-muted-foreground block">
                          Effective Base
                        </span>
                        <span className="font-mono font-semibold text-foreground">
                          {(
                            result.prediction.explanations.effective_baseline_probability * 100
                          ).toFixed(1)}
                          %
                        </span>
                      </div>
                    </div>

                    {/* Factors list */}
                    <div className="space-y-2 pt-1">
                      <span className="text-[11px] font-semibold text-slate-300 block">
                        Significant Feature Drivers:
                      </span>
                      {result.prediction.explanations.factors &&
                      result.prediction.explanations.factors.length > 0 ? (
                        <div className="space-y-2">
                          {result.prediction.explanations.factors.map((factor) => {
                            const isIncrease = factor.direction === "increases_risk";
                            return (
                              <div
                                key={factor.feature}
                                className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 rounded-md border border-border/60 bg-background/60 p-2 text-xs"
                              >
                                <div className="space-y-0.5">
                                  <div className="flex items-center gap-2">
                                    <span className="font-semibold text-foreground">
                                      {factor.feature}
                                    </span>
                                    <span className="rounded bg-elevated px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground border border-border/40">
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
                                      isIncrease ? "text-red-400" : "text-emerald-400"
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
                                    className={`text-[9px] font-semibold uppercase tracking-wider px-1.5 py-0.2 rounded border ${
                                      isIncrease
                                        ? "border-red-500/30 bg-red-950/30 text-red-300"
                                        : "border-emerald-500/30 bg-emerald-950/30 text-emerald-300"
                                    }`}
                                  >
                                    {isIncrease ? "Raises Risk" : "Lowers Risk"}
                                  </span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <p className="rounded-md border border-dashed border-border/80 p-3 text-center text-xs text-muted-foreground italic">
                          No active input shifted model probability beyond the ±1.0% threshold.
                        </p>
                      )}
                    </div>

                    <p className="text-[10px] text-muted-foreground leading-relaxed italic border-t border-border/40 pt-2">
                      {result.prediction.explanations.disclaimer}
                    </p>
                  </div>
                )}
              </div>
            ) : result?.status === "error" ? (
              <div className="rounded-lg border border-red-500/30 bg-red-950/20 p-4 text-red-300">
                <p className="font-semibold text-xs sm:text-sm">Inference Service Error</p>
                <p className="mt-1 text-xs text-red-200/80">{result.message}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 py-16 text-center text-muted-foreground">
                <Sparkles className="size-8 opacity-30 text-primary" />
                <p className="text-xs font-medium text-slate-300">
                  Awaiting Transaction Submission
                </p>
                <p className="text-[11px] text-muted-foreground max-w-xs">
                  Fill in parameters on the left and trigger evaluation to view real-time model
                  decision tree attributions.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
