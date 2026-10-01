import { createFileRoute } from "@tanstack/react-router";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, ShieldCheck, AlertTriangle, Percent } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { PageHeader, DemoBanner } from "@/components/PageHeader";
import {
  DEMO_NOTICE,
  demoDistribution,
  demoFraudRate,
  demoTotals,
  demoTransactions,
  demoTrend,
  formatAmount,
  formatTimestamp,
} from "@/lib/demo-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — FraudLens Fraud Detection Prototype" },
      {
        name: "description",
        content:
          "Overview dashboard with synthetic transaction totals, trend and distribution charts for an academic fraud detection prototype.",
      },
      { property: "og:title", content: "Dashboard — FraudLens Fraud Detection Prototype" },
      {
        property: "og:description",
        content: "Synthetic transaction totals, trends and distribution for an academic prototype.",
      },
    ],
  }),
  component: Dashboard,
});

const summary = [
  {
    label: "Total Transactions",
    value: demoTotals.total.toString(),
    icon: Activity,
    tone: "text-primary",
  },
  {
    label: "Legitimate Transactions",
    value: demoTotals.legitimate.toString(),
    icon: ShieldCheck,
    tone: "text-success",
  },
  {
    label: "Potential Fraud",
    value: demoTotals.flagged.toString(),
    icon: AlertTriangle,
    tone: "text-warning",
  },
  {
    label: "Fraud Rate",
    value: `${demoFraudRate}%`,
    icon: Percent,
    tone: "text-destructive",
  },
];

const pieColors = ["var(--chart-1)", "var(--chart-4)"];

function Dashboard() {
  const recent = demoTransactions.slice(0, 8);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
      <PageHeader
        title="Dashboard"
        description="Prototype overview of sample transaction activity."
      />
      <DemoBanner text={DEMO_NOTICE} />

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {summary.map(({ label, value, icon: Icon, tone }) => (
          <Card key={label} className="shadow-card">
            <CardContent className="flex items-center justify-between gap-4 p-5">
              <div className="min-w-0">
                <p className="truncate text-sm text-muted-foreground">{label}</p>
                <p className="mt-2 text-3xl font-semibold tracking-tight">{value}</p>
                <p className="mt-1 text-xs text-muted-foreground">Sample data</p>
              </div>
              <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-accent">
                <Icon className={`size-5 ${tone}`} aria-hidden />
              </span>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="shadow-card lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-base">Transaction trend (sample)</CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={demoTrend} margin={{ left: -16, right: 8, top: 8 }}>
                <defs>
                  <linearGradient id="txnFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--chart-1)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="var(--chart-1)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="day" tickLine={false} axisLine={false} fontSize={12} />
                <YAxis tickLine={false} axisLine={false} fontSize={12} />
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid var(--border)",
                    background: "var(--card)",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="transactions"
                  stroke="var(--chart-1)"
                  strokeWidth={2}
                  fill="url(#txnFill)"
                  name="Transactions"
                />
                <Area
                  type="monotone"
                  dataKey="flagged"
                  stroke="var(--chart-4)"
                  strokeWidth={2}
                  fill="none"
                  name="Flagged"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="shadow-card">
          <CardHeader>
            <CardTitle className="text-base">Label distribution (sample)</CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={demoDistribution}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={3}
                >
                  {demoDistribution.map((entry, i) => (
                    <Cell key={entry.name} fill={pieColors[i % pieColors.length]} />
                  ))}
                </Pie>
                <Legend verticalAlign="bottom" iconType="circle" />
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: "1px solid var(--border)",
                    background: "var(--card)",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card className="shadow-card">
        <CardHeader>
          <CardTitle className="text-base">Recent transactions (sample)</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Transaction ID</TableHead>
                <TableHead>Amount</TableHead>
                <TableHead>Timestamp (UTC)</TableHead>
                <TableHead>Channel</TableHead>
                <TableHead>Sample label</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recent.map((t) => (
                <TableRow key={t.id}>
                  <TableCell className="font-mono text-xs">{t.id}</TableCell>
                  <TableCell>{formatAmount(t.amount)}</TableCell>
                  <TableCell className="text-muted-foreground">
                    {formatTimestamp(t.timestamp)}
                  </TableCell>
                  <TableCell>{t.channel}</TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={
                        t.label === "Flagged (sample)"
                          ? "border-warning-border bg-warning-surface text-warning-foreground"
                          : "border-transparent bg-success-surface text-success-foreground"
                      }
                    >
                      {t.label}
                    </Badge>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
