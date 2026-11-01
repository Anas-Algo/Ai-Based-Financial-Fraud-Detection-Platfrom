import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
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
import {
  Activity,
  ShieldCheck,
  AlertTriangle,
  Percent,
  ArrowRight,
  Database,
  Loader2,
  Sparkles,
  RefreshCw,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
  statsQueryOptions,
  transactionsQueryOptions,
  analyticsQueryOptions,
  healthQueryOptions,
} from "@/lib/api-queries";
import { isApiConfigured } from "@/lib/fraud-api";
import {
  DEMO_NOTICE,
  demoDistribution,
  demoFraudRate,
  demoTotals,
  demoTransactions,
  demoTrend,
  formatAmount,
  formatTimestamp,
  formatTransactionAmount,
} from "@/lib/demo-data";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — FraudLens AI Detection" },
      {
        name: "description",
        content:
          "Overview dashboard showing real-time AI transaction statistics, fraud rate, and recent predictions from the Random Forest model.",
      },
      { property: "og:title", content: "Dashboard — FraudLens AI Detection" },
      {
        property: "og:description",
        content:
          "Real-time AI fraud detection overview with live metrics and recent model predictions.",
      },
    ],
  }),
  component: Dashboard,
});

const pieColors = ["#10b981", "#ef4444"];

function Dashboard() {
  const [showDemoFallback, setShowDemoFallback] = useState(false);

  const apiConfigured = isApiConfigured();
  const {
    data: health,
    isError: healthError,
    refetch: refetchHealth,
  } = useQuery(healthQueryOptions());
  const {
    data: stats,
    isLoading: statsLoading,
    isError: statsError,
    isFetching: statsFetching,
    refetch: refetchStats,
  } = useQuery(statsQueryOptions());
  const {
    data: txData,
    isLoading: txLoading,
    isError: txError,
    isFetching: txFetching,
    refetch: refetchTx,
  } = useQuery(transactionsQueryOptions(8));
  const {
    data: analytics,
    isLoading: anLoading,
    isError: anError,
    isFetching: anFetching,
    refetch: refetchAnalytics,
  } = useQuery(analyticsQueryOptions());

  const isFetching = statsFetching || txFetching || anFetching;
  const isBackendError = Boolean(healthError || statsError);
  const handleRefresh = () => {
    refetchHealth();
    refetchStats();
    refetchTx();
    refetchAnalytics();
  };

  const hasLiveTransactions = Boolean(stats && stats.totalTransactions > 0);
  const useLive = apiConfigured && !isBackendError && (hasLiveTransactions || !showDemoFallback);

  const total = useLive
    ? (stats?.totalTransactions ?? 0)
    : showDemoFallback || !apiConfigured
      ? demoTotals.total
      : 0;
  const legitimate = useLive
    ? (stats?.legitimateTransactions ?? 0)
    : showDemoFallback || !apiConfigured
      ? demoTotals.legitimate
      : 0;
  const flagged = useLive
    ? (stats?.fraudulentTransactions ?? 0)
    : showDemoFallback || !apiConfigured
      ? demoTotals.flagged
      : 0;
  const fraudRate = useLive
    ? (stats?.fraudRate ?? 0)
    : showDemoFallback || !apiConfigured
      ? demoFraudRate
      : 0;

  const kpiNote = (liveNote: string) => {
    if (isBackendError && !showDemoFallback) return "Backend unreachable";
    if (useLive) return liveNote;
    return "Synthetic sample";
  };

  const summary = [
    {
      label: "Total Transactions",
      value: isBackendError && !showDemoFallback ? "—" : total.toString(),
      icon: Activity,
      tone: "text-primary",
      note: kpiNote("Live database"),
    },
    {
      label: "Legitimate Transactions",
      value: isBackendError && !showDemoFallback ? "—" : legitimate.toString(),
      icon: ShieldCheck,
      tone: "text-emerald-600",
      note: kpiNote("Model predicted legitimate"),
    },
    {
      label: "Fraudulent Transactions",
      value: isBackendError && !showDemoFallback ? "—" : flagged.toString(),
      icon: AlertTriangle,
      tone: "text-destructive",
      note: kpiNote("Model flagged fraud"),
    },
    {
      label: "Fraud Rate",
      value: isBackendError && !showDemoFallback ? "—" : `${fraudRate}%`,
      icon: Percent,
      tone: "text-destructive",
      note: kpiNote("Current audit percentage"),
    },
  ];

  type DisplayRow = {
    id: string;
    amount: number;
    currency?: string | null;
    timestamp: string;
    createdAt?: string;
    hasTransactionTime?: boolean;
    channel: string;
    label: string;
    riskLevel?: string;
  };

  const recentLive = txData?.transactions ?? [];
  const recentDemo = demoTransactions.slice(0, 8);

  const displayRows: DisplayRow[] =
    useLive && hasLiveTransactions
      ? recentLive.map((t) => ({
          id: t.transactionId,
          amount: t.amount,
          currency: t.currency,
          timestamp: t.timestamp,
          createdAt: t.createdAt,
          hasTransactionTime: t.hasTransactionTime,
          channel: t.channel,
          label: t.label,
          riskLevel: t.riskLevel,
        }))
      : showDemoFallback || !apiConfigured
        ? recentDemo.map((t) => ({
            id: t.id,
            amount: t.amount,
            currency: "USD",
            timestamp: t.timestamp,
            channel: t.channel,
            label: t.label,
            riskLevel: t.label === "Flagged (sample)" ? "High" : "Low",
          }))
        : [];

  // Distribution chart data
  const distributionData = useLive
    ? analytics?.labelDistribution &&
      (analytics.labelDistribution[0]?.value > 0 || analytics.labelDistribution[1]?.value > 0)
      ? analytics.labelDistribution
      : [
          { name: "Legitimate", value: legitimate },
          { name: "Fraudulent", value: flagged },
        ]
    : showDemoFallback || !apiConfigured
      ? demoDistribution
      : [
          { name: "Legitimate", value: 0 },
          { name: "Fraudulent", value: 0 },
        ];

  // Trend chart data
  const trendData =
    useLive && analytics?.transactionTrend && analytics.transactionTrend.length > 0
      ? analytics.transactionTrend
      : showDemoFallback || !apiConfigured
        ? demoTrend
        : [];

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
      <PageHeader
        title="Dashboard"
        description="Live overview of transaction fraud detection metrics and recent AI model evaluations."
      >
        <div className="flex items-center gap-2">
          {apiConfigured && (
            <Badge
              variant="outline"
              className={
                health?.status === "healthy" || (useLive && hasLiveTransactions)
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                  : "border-muted"
              }
            >
              <Database className="size-3 mr-1" />
              {health?.status === "healthy"
                ? "Flask API Connected"
                : useLive && hasLiveTransactions
                  ? "Live Flask API"
                  : "Backend Connected"}
            </Badge>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isFetching}
            className="h-8 text-xs"
          >
            <RefreshCw className={`size-3.5 mr-1 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </PageHeader>

      {/* Connection & Mode Banner */}
      {!apiConfigured ? (
        <DemoBanner text="ML Backend is not configured. Displaying synthetic demonstration data." />
      ) : isBackendError ? (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-xl border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          <div className="flex items-center gap-2">
            <AlertTriangle className="size-4 shrink-0" />
            <span>
              Could not connect to Flask API on{" "}
              <code className="font-mono">http://127.0.0.1:5000</code>. Please check that the
              backend service is running.
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              className="h-8 text-xs border-destructive/30 hover:bg-destructive/20"
            >
              <RefreshCw className="size-3 mr-1" />
              Retry Connection
            </Button>
          </div>
        </div>
      ) : hasLiveTransactions ? (
        <div className="flex items-center justify-between rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-800 dark:text-emerald-200">
          <div className="flex items-center gap-2">
            <ShieldCheck className="size-4 shrink-0 text-emerald-600" />
            <span>Showing live metrics calculated from the SQLite transaction audit log.</span>
          </div>
          <span className="font-mono text-xs">{stats?.totalTransactions} analyzed</span>
        </div>
      ) : (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm">
          <div className="flex items-center gap-2">
            <Sparkles className="size-4 text-primary shrink-0" />
            <span>
              Connected to Flask model service ({health?.model_name || "Random Forest"}). Database
              is ready for transactions.
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDemoFallback(!showDemoFallback)}
              className="h-8 text-xs"
            >
              {showDemoFallback ? "Hide Sample Data" : "Preview with Sample Data"}
            </Button>
            <Button asChild size="sm" className="h-8 text-xs">
              <Link to="/analyze">
                Analyze a Transaction <ArrowRight className="size-3 ml-1" />
              </Link>
            </Button>
          </div>
        </div>
      )}

      {/* Summary KPI Cards */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {summary.map(({ label, value, icon: Icon, tone, note }) => (
          <Card key={label} className="shadow-card">
            <CardContent className="flex items-center justify-between gap-4 p-5">
              <div className="min-w-0">
                <p className="truncate text-sm text-muted-foreground">{label}</p>
                <p className="mt-2 text-3xl font-semibold tracking-tight">
                  {statsLoading ? <Loader2 className="size-6 animate-spin" /> : value}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">{note}</p>
              </div>
              <span className="grid size-11 shrink-0 place-items-center rounded-xl bg-accent">
                <Icon className={`size-5 ${tone}`} aria-hidden />
              </span>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Trend Chart */}
        <Card className="shadow-card lg:col-span-2">
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-base">Transaction Volume & Fraud Trend</CardTitle>
            <Badge variant="outline" className="text-xs font-normal">
              {useLive && hasLiveTransactions ? "Live Log" : "Sample Trend"}
            </Badge>
          </CardHeader>
          <CardContent className="h-72">
            {trendData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData} margin={{ left: -16, right: 8, top: 8 }}>
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
                    stroke="#ef4444"
                    strokeWidth={2}
                    fill="none"
                    name="Fraud Flagged"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
                <Activity className="size-8 opacity-40" />
                <p className="text-sm">No transaction trend data recorded yet.</p>
                <p className="text-xs">Analyzed transactions will appear plotted here by date.</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Label Distribution Chart */}
        <Card className="shadow-card">
          <CardHeader className="flex flex-row items-center justify-between space-y-0">
            <CardTitle className="text-base">Prediction Distribution</CardTitle>
            <Badge variant="outline" className="text-xs font-normal">
              {useLive && hasLiveTransactions ? "Live Ratio" : "Sample Ratio"}
            </Badge>
          </CardHeader>
          <CardContent className="h-72">
            {total > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={distributionData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={3}
                  >
                    {distributionData.map((entry, i) => (
                      <Cell
                        key={entry.name}
                        fill={entry.name.toLowerCase().includes("fraud") ? "#ef4444" : "#10b981"}
                      />
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
            ) : (
              <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
                <ShieldCheck className="size-8 opacity-40" />
                <p className="text-sm">No predictions yet.</p>
                <p className="text-xs">Class distribution will render here.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Transactions Table */}
      <Card className="shadow-card">
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-base">Recent Transactions</CardTitle>
            <p className="text-xs text-muted-foreground mt-1">
              {useLive && hasLiveTransactions
                ? "Latest transactions evaluated by the machine learning pipeline."
                : showDemoFallback || !apiConfigured
                  ? "Synthetic transactions for preview."
                  : isBackendError
                    ? "Live audit records could not be loaded from the backend."
                    : "No transactions recorded yet."}
            </p>
          </div>
          <Button variant="ghost" size="sm" asChild className="text-xs">
            <Link to="/history">
              View All History <ArrowRight className="size-3 ml-1" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          {txLoading ? (
            <div className="flex flex-col items-center justify-center gap-2 py-10 text-muted-foreground">
              <Loader2 className="size-6 animate-spin text-primary" />
              <p className="text-xs">Loading recent transactions…</p>
            </div>
          ) : (txError || isBackendError) && !showDemoFallback ? (
            <div className="flex flex-col items-center justify-center gap-3 py-10 text-center text-muted-foreground">
              <AlertTriangle className="size-8 text-destructive opacity-80" />
              <p className="text-sm font-medium text-destructive">
                Unable to load recent transactions from the Flask backend.
              </p>
              <Button variant="outline" size="sm" onClick={() => refetchTx()} className="text-xs">
                <RefreshCw className="size-3 mr-1" />
                Retry
              </Button>
            </div>
          ) : useLive && !hasLiveTransactions && !showDemoFallback ? (
            <div className="flex flex-col items-center justify-center gap-3 py-10 text-center text-muted-foreground">
              <Database className="size-8 opacity-40" />
              <p className="text-sm font-medium">
                No transactions recorded in the audit database yet.
              </p>
              <Button asChild size="sm">
                <Link to="/analyze">Analyze First Transaction</Link>
              </Button>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Transaction ID</TableHead>
                  <TableHead>Amount</TableHead>
                  <TableHead>Transaction / Analysis Time</TableHead>
                  <TableHead>Channel</TableHead>
                  <TableHead>Prediction</TableHead>
                  <TableHead>Risk Level</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {displayRows.map((t) => (
                  <TableRow key={t.id}>
                    <TableCell className="font-mono text-xs">{t.id}</TableCell>
                    <TableCell>
                      {(() => {
                        const formatted = formatTransactionAmount(t.amount, t.currency);
                        return (
                          <div className="flex flex-col sm:flex-row sm:items-center gap-1.5">
                            <span className="font-medium">{formatted.text}</span>
                            {formatted.isLegacy && (
                              <Badge
                                variant="outline"
                                className="w-fit text-[9px] font-normal border-dashed text-muted-foreground"
                              >
                                Legacy / Unspecified
                              </Badge>
                            )}
                          </div>
                        );
                      })()}
                    </TableCell>
                    <TableCell className="text-xs">
                      {t.hasTransactionTime ? (
                        <div className="space-y-0.5">
                          <div>
                            <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider block">
                              Transaction Time
                            </span>
                            <span className="font-medium text-foreground">
                              {formatTimestamp(t.timestamp)}
                            </span>
                          </div>
                          {t.createdAt ? (
                            <div>
                              <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider block">
                                Analyzed At
                              </span>
                              <span className="text-muted-foreground">
                                {formatTimestamp(t.createdAt)}
                              </span>
                            </div>
                          ) : null}
                        </div>
                      ) : (
                        <div className="space-y-0.5">
                          <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider block">
                            Recorded / Analysis Time
                          </span>
                          <span className="text-foreground">{formatTimestamp(t.timestamp)}</span>
                        </div>
                      )}
                    </TableCell>
                    <TableCell>{t.channel}</TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={
                          t.label === "Fraudulent" || t.label === "Flagged (sample)"
                            ? "border-destructive/30 bg-destructive/10 text-destructive font-semibold"
                            : "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 font-semibold"
                        }
                      >
                        {t.label}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant="outline"
                        className={
                          t.riskLevel === "High"
                            ? "border-destructive/30 bg-destructive/10 text-destructive font-medium"
                            : t.riskLevel === "Medium"
                              ? "border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium"
                              : "border-emerald-500/30 bg-emerald-500/10 text-emerald-600 font-medium"
                        }
                      >
                        {t.riskLevel ||
                          (t.label.includes("Fraud") || t.label.includes("Flagged")
                            ? "High"
                            : "Low")}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
