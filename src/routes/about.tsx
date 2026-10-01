import { createFileRoute } from "@tanstack/react-router";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader, DemoBanner } from "@/components/PageHeader";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — FraudLens Academic Prototype" },
      {
        name: "description",
        content:
          "Academic purpose, planned methodology, dataset intent, model plan and limitations of this fraud detection prototype.",
      },
      { property: "og:title", content: "About — FraudLens Academic Prototype" },
      {
        property: "og:description",
        content: "Purpose, methodology, dataset plan and limitations of this student prototype.",
      },
    ],
  }),
  component: AboutPage,
});

const sections = [
  {
    title: "Academic purpose",
    body: "FraudLens is a third-semester computer science coursework project. It demonstrates how a fraud-detection analyst interface could be structured. It is not a banking product and is not used with real customer data.",
  },
  {
    title: "Methodology (planned)",
    body: "The intended workflow is: collect a labelled transaction dataset, clean and encode the features, train a supervised classifier offline in Python, expose it through a Flask REST API, and let this frontend send transaction details to that API for a prediction. Only the frontend stage is complete.",
  },
  {
    title: "Dataset",
    body: "No dataset is bundled with this app. Every number, chart and table you see is synthetic demonstration data generated in the frontend so the layout can be reviewed. A public labelled dataset is planned for the training stage.",
  },
  {
    title: "ML model",
    body: "No model is trained, loaded or connected. The Analyze page validates your input and then reports 'ML model not connected yet'. No risk score, probability or accuracy figure is produced anywhere in this build.",
  },
  {
    title: "Limitations",
    body: "No database, no authentication, no live transaction feed, and no connection to banks, UPI or payment gateways. Results shown must not be interpreted as real fraud detection outcomes.",
  },
  {
    title: "Planned integration",
    body: "The frontend keeps a single integration seam for the backend. When the Flask service is ready, its base URL is configured in one place and the Analyze page will display the returned prediction instead of the not-connected notice.",
  },
];

function AboutPage() {
  return (
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-6">
      <PageHeader
        title="About this project"
        description="Scope, method and honest limitations of the prototype."
      />
      <DemoBanner text="Educational prototype — no trained model, dataset, database or real transactions are connected." />

      <div className="grid gap-4">
        {sections.map((s) => (
          <Card key={s.title} className="shadow-card">
            <CardHeader>
              <CardTitle className="text-base">{s.title}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm leading-relaxed text-muted-foreground">
              {s.body}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
