import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Loader2, PlugZap } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { PageHeader, DemoBanner } from "@/components/PageHeader";
import { analyzeTransaction, type PredictionResult } from "@/lib/fraud-api";

export const Route = createFileRoute("/analyze")({
  head: () => ({
    meta: [
      { title: "Analyze Transaction — FraudLens Prototype" },
      {
        name: "description",
        content:
          "Enter sample transaction details to see where a future Flask machine-learning service will return a prediction.",
      },
      { property: "og:title", content: "Analyze Transaction — FraudLens Prototype" },
      {
        property: "og:description",
        content: "Sample transaction form prepared for a future Flask prediction service.",
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
  originCountry: string;
  accountAgeDays: string;
};

const initialForm: FormState = {
  transactionId: "",
  amount: "",
  channel: "Card",
  merchantCategory: "",
  originCountry: "",
  accountAgeDays: "",
};

const channels = ["Card", "Transfer", "Wallet", "ATM"];

function validate(form: FormState) {
  const errors: Partial<Record<keyof FormState, string>> = {};
  if (!form.transactionId.trim()) errors.transactionId = "Transaction ID is required.";
  const amount = Number(form.amount);
  if (!form.amount.trim()) errors.amount = "Amount is required.";
  else if (Number.isNaN(amount) || amount <= 0) errors.amount = "Enter an amount greater than 0.";
  if (!form.merchantCategory.trim()) errors.merchantCategory = "Merchant category is required.";
  if (!form.originCountry.trim()) errors.originCountry = "Origin country is required.";
  const age = Number(form.accountAgeDays);
  if (!form.accountAgeDays.trim()) errors.accountAgeDays = "Account age is required.";
  else if (Number.isNaN(age) || age < 0) errors.accountAgeDays = "Enter 0 or more days.";
  return errors;
}

function AnalyzePage() {
  const [form, setForm] = useState<FormState>(initialForm);
  const [errors, setErrors] = useState<Partial<Record<keyof FormState, string>>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const set = (key: keyof FormState) => (value: string) =>
    setForm((prev) => ({ ...prev, [key]: value }));

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
      transactionId: form.transactionId,
      amount: Number(form.amount),
      channel: form.channel,
      merchantCategory: form.merchantCategory,
      originCountry: form.originCountry,
      accountAgeDays: Number(form.accountAgeDays),
    });
    setResult(response);
    setLoading(false);
  }

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
      <PageHeader
        title="Analyze Transaction"
        description="Submit sample transaction details through the prototype form."
      />
      <DemoBanner text="No machine-learning model is connected. Submitting the form will not produce a fraud prediction." />

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <Card className="shadow-card">
          <CardHeader>
            <CardTitle className="text-base">Sample transaction details</CardTitle>
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
                label="Amount"
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
              <Field
                id="merchantCategory"
                label="Merchant category"
                value={form.merchantCategory}
                onChange={set("merchantCategory")}
                error={errors.merchantCategory}
                placeholder="Electronics"
              />
              <Field
                id="originCountry"
                label="Origin country"
                value={form.originCountry}
                onChange={set("originCountry")}
                error={errors.originCountry}
                placeholder="India"
              />
              <Field
                id="accountAgeDays"
                label="Account age (days)"
                type="number"
                value={form.accountAgeDays}
                onChange={set("accountAgeDays")}
                error={errors.accountAgeDays}
                placeholder="420"
              />

              <div className="flex flex-wrap gap-3 sm:col-span-2">
                <Button type="submit" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 className="size-4 animate-spin" aria-hidden /> Analyzing…
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
                >
                  Reset
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        <Card className="shadow-card">
          <CardHeader>
            <CardTitle className="text-base">Prediction result</CardTitle>
          </CardHeader>
          <CardContent aria-live="polite" className="text-sm">
            {loading ? (
              <p className="flex items-center gap-2 text-muted-foreground">
                <Loader2 className="size-4 animate-spin" aria-hidden /> Contacting prediction
                service…
              </p>
            ) : result?.status === "not_connected" ? (
              <div className="rounded-xl border border-warning-border bg-warning-surface p-4">
                <p className="flex items-center gap-2 font-semibold text-warning-foreground">
                  <PlugZap className="size-4" aria-hidden /> {result.message}
                </p>
                <p className="mt-2 text-warning-foreground/90">
                  The form input was validated, but no prediction can be produced until the Python
                  Flask REST API is integrated. No risk score or label is generated here.
                </p>
              </div>
            ) : result?.status === "error" ? (
              <p className="rounded-xl bg-destructive/10 p-4 text-destructive">{result.message}</p>
            ) : (
              <p className="text-muted-foreground">
                Fill in the sample details and select Analyze. Results will appear here once the
                backend model is connected.
              </p>
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
  error?: string;
  placeholder?: string;
  type?: string;
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
