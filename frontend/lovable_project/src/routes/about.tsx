import { createFileRoute } from "@tanstack/react-router";
import {
  Brain,
  Cpu,
  Database,
  Network,
  ShieldAlert,
  Sparkles,
  Terminal,
  Code2,
  Binary,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PageHeader } from "@/components/PageHeader";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "Methodology & Architecture — FraudLens" },
      {
        name: "description",
        content:
          "Machine learning pipeline architecture, dataset methodology, REST API contracts, and academic prototype disclosures of FraudLens.",
      },
      { property: "og:title", content: "Methodology & Architecture — FraudLens" },
      {
        property: "og:description",
        content: "Purpose, ML methodology, dataset, and architecture of the FraudLens platform.",
      },
    ],
  }),
  component: AboutPage,
});

const specs = [
  { label: "Classifier Architecture", value: "Random Forest (80 Decision Trees)", icon: Binary },
  { label: "Target Domain Rail", value: "Indian UPI & Digital Payments", icon: Cpu },
  { label: "Active Base Currency", value: "Indian Rupee (INR / ₹)", icon: Database },
  { label: "Temporal Frame", value: "IST (UTC+05:30) Synchronized", icon: Network },
];

const sections = [
  {
    icon: Brain,
    title: "Platform Mission & Architecture",
    body: "FraudLens is an academic demonstration prototype exploring AI-based financial fraud detection. It connects a scikit-learn decision-tree ensemble with a local Flask REST service, persistent SQLite audit logging, and an interactive evaluation dashboard. It is an educational research system and is not connected to real banking switches.",
  },
  {
    icon: Cpu,
    title: "Machine Learning Pipeline & Features",
    body: "The core engine features an 80-tree Random Forest classifier calibrated on financial transaction features using synthetic benchmark datasets: transaction amount (INR), transaction type (P2M vs P2P), merchant category (including Education, Groceries, Quick Commerce, Dining, Travel), and temporal signals derived in India Standard Time (hour of day, day of month, calendar month, day of week, weekend indicator). Balanced class weighting addresses the extreme rarity of fraudulent events in synthetic training data.",
  },
  {
    icon: Database,
    title: "Data Generation & Model Artifacts",
    body: "The model is serialized as a lightweight joblib pipeline (fraudlens_india_pipeline.pkl) containing complete preprocessors, one-hot encoders, and tree structures. Synthetic benchmarks model realistic Indian payment behavior, transaction values, and payment channels for evaluation.",
  },
  {
    icon: Network,
    title: "Flask REST API & SQLite Audit Log",
    body: "The Flask server operates locally on port 5000 with hardened validation schemas. Transactions evaluated via /api/predict are stamped with timestamps and written to a SQLite database (transactions.db). Summary statistics (/api/stats) and visual breakdowns (/api/analytics) are calculated directly from persistent audit rows.",
  },
  {
    icon: Sparkles,
    title: "Frontend Engineering & UI System",
    body: "Constructed with React 19, TanStack Start, TypeScript, Vite, Tailwind CSS v4, and Recharts. Built with strict type validation, accessible components, and a dedicated dark-theme fintech interface for evaluating simulated transaction risk.",
  },
  {
    icon: ShieldAlert,
    title: "Academic Prototype Disclosures & Limits",
    body: "FraudLens is an academic research and demonstration prototype. It does not interface with live banking switches (NPCI UPI switches, SWIFT, Visa/Mastercard networks) and is not validated for production fraud prevention. Decision tree attributions represent statistical path contributions relative to baseline probability and do not constitute legal proof of fraudulent intent.",
  },
];

export function AboutPage() {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
      <PageHeader
        title="System Methodology & Architecture"
        description="Technical specifications of the academic machine learning pipeline, local inference server, and audit persistence layer."
      />

      {/* Hero Overview Card */}
      <div className="rounded-lg border border-primary/30 bg-primary/5 p-5 text-sm leading-relaxed text-slate-300">
        <div className="flex items-center gap-2 mb-2">
          <Badge className="bg-primary/20 text-primary border-primary/30 text-[10px] font-mono uppercase tracking-wider">
            Architecture Specification
          </Badge>
          <span className="text-xs text-muted-foreground font-mono">v5.0-fintech-dark</span>
        </div>
        <p className="text-slate-300 text-xs sm:text-sm">
          <strong className="text-foreground">FraudLens</strong> is an academic machine learning
          prototype for fraud detection research, anomaly attribution, and risk scoring. Combining
          calibrated decision-tree ensembles with persistent audit logging, the platform
          demonstrates how AI models evaluate simulated transactions across Indian digital payment
          rails using synthetic benchmark data. It is an educational prototype and is not connected
          to real banking switches or certified for production fraud prevention.
        </p>
      </div>

      {/* Technical Specifications Grid */}
      <div className="grid gap-3.5 sm:grid-cols-2 lg:grid-cols-4">
        {specs.map(({ label, value, icon: Icon }) => (
          <div
            key={label}
            className="flex flex-col justify-between gap-2.5 rounded-lg border border-border/80 bg-card p-4 shadow-card"
          >
            <div className="flex items-center gap-2.5">
              <div className="grid size-7 shrink-0 place-items-center rounded-md border border-primary/20 bg-primary/10 text-primary">
                <Icon className="size-3.5" />
              </div>
              <span className="text-[10px] uppercase font-semibold tracking-wider text-muted-foreground leading-normal">
                {label}
              </span>
            </div>
            <div className="pt-1 border-t border-border/40">
              <span className="font-mono text-xs font-semibold text-foreground leading-snug break-words">
                {value}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Detailed Section Cards */}
      <div className="grid gap-4">
        {sections.map(({ icon: Icon, title, body }) => (
          <Card key={title} className="border-border/80 bg-card shadow-card">
            <CardHeader className="flex flex-row items-center gap-3 pb-2 space-y-0">
              <div className="grid size-8 shrink-0 place-items-center rounded-lg border border-border bg-elevated text-primary">
                <Icon className="size-4" />
              </div>
              <CardTitle className="text-sm font-semibold tracking-tight text-foreground">
                {title}
              </CardTitle>
            </CardHeader>
            <CardContent className="text-xs leading-relaxed text-muted-foreground pl-14">
              {body}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
