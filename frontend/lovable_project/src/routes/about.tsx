import { createFileRoute } from "@tanstack/react-router";
import { Brain, Cpu, Database, Network, ShieldAlert, Sparkles } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/PageHeader";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — FraudLens AI Fraud Detection" },
      {
        name: "description",
        content:
          "Academic purpose, machine learning pipeline, dataset information, REST API architecture, and honest limitations of FraudLens.",
      },
      { property: "og:title", content: "About — FraudLens AI Fraud Detection" },
      {
        property: "og:description",
        content: "Purpose, ML methodology, dataset, and architecture of the FraudLens project.",
      },
    ],
  }),
  component: AboutPage,
});

const sections = [
  {
    icon: Brain,
    title: "Project Purpose & Scope",
    body: "FraudLens is a full-stack financial fraud detection system built for coursework demonstration. It provides an end-to-end prototype showing how machine learning inference models can be integrated into analyst dashboards to assess and monitor transaction risks in real time.",
  },
  {
    icon: Cpu,
    title: "Machine Learning Pipeline",
    body: "The core ML engine is built using Scikit-learn. It uses a ColumnTransformer pipeline that applies OneHotEncoding with unknown-category handling to categorical features (merchant category, gender) and median imputation to numeric features (amount, location coordinates, city population, transaction hour/day/month). Inference is performed by a RandomForestClassifier trained with balanced class weights to address severe fraud class imbalance.",
  },
  {
    icon: Database,
    title: "Training Dataset",
    body: "The model was trained offline using the public Kaggle Credit Card Fraud dataset (fraudTrain.csv), which contains simulated credit card transactions covering legitimate and fraudulent patterns. The saved pipeline artifact (fraud_detection_pipeline.pkl) is loaded directly by the backend for rapid local inference.",
  },
  {
    icon: Network,
    title: "Backend REST API & Audit Persistence",
    body: "The backend is implemented as a Python Flask REST service running on port 5000. It provides dedicated endpoints for health checks (/api/health), transaction evaluation (/api/predict), audit history (/api/transactions), summary metrics (/api/stats), and chart aggregations (/api/analytics). All evaluated transactions are persisted locally in a SQLite database (transactions.db) for reliable audit logging.",
  },
  {
    icon: Sparkles,
    title: "Frontend Architecture",
    body: "The analyst interface is built with React 19, TanStack Start, TypeScript, Vite, Tailwind CSS, and Recharts. It communicates with Flask via strongly typed clients and React Query, ensuring automatic cache invalidation and UI synchronization across the Dashboard, Analyze, History, and Analytics pages.",
  },
  {
    icon: ShieldAlert,
    title: "Honest Academic Limitations",
    body: "This project is an academic prototype. It is not connected to live banking rails, payment gateways (UPI, SWIFT, credit networks), or real customer financial accounts. Features required by the offline dataset but omitted from the simplified web input form (such as exact merchant GPS coordinates) use documented neutral defaults. Model confidence scores reflect classifier probability estimates and must not be treated as absolute guarantees.",
  },
];

function AboutPage() {
  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
      <PageHeader
        title="About FraudLens"
        description="Architecture, machine learning methodology, dataset, and honest limitations of this project."
      />

      <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 text-sm text-foreground/90 leading-relaxed">
        <strong>FraudLens — AI-Based Financial Fraud Detection System</strong> demonstrates
        end-to-end integration between a Scikit-learn machine learning pipeline and a modern
        reactive dashboard.
      </div>

      <div className="grid gap-4">
        {sections.map(({ icon: Icon, title, body }) => (
          <Card key={title} className="shadow-card">
            <CardHeader className="flex flex-row items-center gap-3 pb-2 space-y-0">
              <span className="grid size-9 place-items-center rounded-lg bg-primary/10 text-primary">
                <Icon className="size-5" />
              </span>
              <CardTitle className="text-base">{title}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm leading-relaxed text-muted-foreground pl-14">
              {body}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
