import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, useMemo } from "react";
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
  Sparkles,
  RefreshCw,
  TrendingUp,
  PieChart as PieIcon,
  Clock,
  CheckCircle2,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
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
import { MetricCard } from "@/components/MetricCard";
import { StatusBadge } from "@/components/StatusBadge";
import { RiskBadge } from "@/components/RiskBadge";
import { EmptyState } from "@/components/EmptyState";
import { LoadingState } from "@/components/LoadingState";
import {
  statsQueryOptions,
  transactionsQueryOptions,
  analyticsQueryOptions,
  healthQueryOptions,
} from "@/lib/api-queries";
import { isApiConfigured } from "@/lib/fraud-api";
import {
  demoDistribution,
  demoFraudRate,
  demoTotals,
  demoTransactions,
  demoTrend,
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

export function Dashboard() {
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

  type DisplayRow = {
    id: string;
    amount: number;
    currency?: string | null | undefined;
    timestamp: string;
    createdAt?: string | undefined;
    hasTransactionTime?: boolean | undefined;
    channel: string;
    label: string;
    riskLevel?: string | undefined;
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
      ((analytics.labelDistribution[0]?.value ?? 0) > 0 ||
        (analytics.labelDistribution[1]?.value ?? 0) > 0)
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

  // Helper to format a Date into YYYY-MM-DD in India Standard Time (IST, UTC+05:30)
  const getISTDayString = (date: Date): string => {
    return new Intl.DateTimeFormat("en-CA", {
      timeZone: "Asia/Kolkata",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    }).format(date);
  };

  // Dynamically calculate a rolling 30-day reporting window using current date in Asia/Kolkata (IST) as anchor
  const {
    trendData,
    historicalExcludedCount,
    futureExcludedCount,
    trendReportingPeriod,
    windowStartDate,
  } = useMemo(() => {
    // 1. Current date in Asia/Kolkata (IST) as reporting window anchor
    const anchor = getISTDayString(new Date());

    // 2. Start date calculated as exactly 29 calendar days before the anchor date in IST
    const [aYear = 2026, aMonth = 11, aDay = 3] = anchor.split("-").map(Number);
    const anchorUtc = new Date(Date.UTC(aYear, aMonth - 1, aDay, 12, 0, 0));
    const startUtc = new Date(anchorUtc.getTime() - 29 * 24 * 60 * 60 * 1000);
    const startDate = getISTDayString(startUtc);
    const reportingPeriod = `${startDate} to ${anchor}`;

    if (!useLive || !analytics?.transactionTrend || analytics.transactionTrend.length === 0) {
      return {
        trendData: showDemoFallback || !apiConfigured ? demoTrend : [],
        historicalExcludedCount: 0,
        futureExcludedCount: 0,
        trendReportingPeriod: showDemoFallback || !apiConfigured ? "Demo Window" : reportingPeriod,
        windowStartDate: startDate,
      };
    }

    // 3. Sort all recorded transaction trend items chronologically
    const allItems = [...analytics.transactionTrend].sort((a, b) => a.day.localeCompare(b.day));

    // 4. Filter actual transaction trend data to this inclusive date range [startDate, anchor] (30 calendar dates)
    const recentItems = allItems.filter((item) => item.day >= startDate && item.day <= anchor);

    // 5. Count historical transactions prior to startDate (< startDate)
    const historicalExcluded = allItems
      .filter((item) => item.day < startDate)
      .reduce((sum, item) => sum + item.transactions, 0);

    // 6. Handle future-dated records safely (> anchor)
    const futureExcluded = allItems
      .filter((item) => item.day > anchor)
      .reduce((sum, item) => sum + item.transactions, 0);

    return {
      trendData: recentItems,
      historicalExcludedCount: historicalExcluded,
      futureExcludedCount: futureExcluded,
      trendReportingPeriod: reportingPeriod,
      windowStartDate: startDate,
    };
  }, [useLive, analytics?.transactionTrend, showDemoFallback, apiConfigured]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
      <PageHeader
        title="Command Center"
        description="Interactive evaluation cockpit, fraud rate analytics, and model telemetry for simulated transactions."
      >
        <div className="flex items-center gap-2.5">
          {apiConfigured && (
            <Badge
              variant="outline"
              className={
                health?.status === "healthy" || (useLive && hasLiveTransactions)
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400 font-medium"
                  : "border-border bg-card text-muted-foreground"
              }
            >
              <Database className="size-3 mr-1 text-emerald-400" />
              {health?.status === "healthy"
                ? "Flask API Live"
                : useLive && hasLiveTransactions
                  ? "Audit DB Connected"
                  : "Backend Connected"}
            </Badge>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isFetching}
            className="h-8 text-xs border-border bg-card hover:bg-elevated text-slate-300"
          >
            <RefreshCw className={`size-3.5 mr-1.5 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </PageHeader>

      {/* Connection & Mode Banner */}
      {!apiConfigured ? (
        <DemoBanner text="ML Backend is not configured. Displaying synthetic demonstration data." />
      ) : isBackendError ? (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-lg border border-destructive/30 bg-destructive/10 px-4 py-3 text-sm text-red-300">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="size-4 shrink-0 text-red-400" />
            <span>
              Could not connect to Flask API on{" "}
              <code className="font-mono text-xs bg-red-950/60 px-1.5 py-0.5 rounded border border-red-800/40 text-red-200">
                http://127.0.0.1:5000
              </code>
              . Verify that the backend service is running.
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              className="h-8 text-xs border-destructive/40 bg-red-950/30 hover:bg-destructive/20 text-red-200"
            >
              <RefreshCw className="size-3 mr-1" />
              Retry Connection
            </Button>
          </div>
        </div>
      ) : hasLiveTransactions ? (
        <div className="flex items-center justify-between rounded-lg border border-emerald-500/25 bg-emerald-500/5 px-4 py-3 text-sm text-emerald-300">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="size-4 shrink-0 text-emerald-400" />
            <span className="text-xs sm:text-sm">
              Live Flask inference service connected. Metrics synced with local SQLite audit
              repository.
            </span>
          </div>
          <span className="font-mono text-xs px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 font-semibold">
            {stats?.totalTransactions} analyzed
          </span>
        </div>
      ) : (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-sm text-slate-300">
          <div className="flex items-center gap-2.5">
            <Sparkles className="size-4 text-primary shrink-0" />
            <span className="text-xs sm:text-sm">
              Connected to Flask API ({health?.model_name || "Random Forest"}). Database is
              initialized for transaction evaluation.
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowDemoFallback(!showDemoFallback)}
              className="h-8 text-xs border-border bg-card hover:bg-elevated text-slate-300"
            >
              {showDemoFallback ? "Hide Sample Data" : "Preview Sample Data"}
            </Button>
            <Button
              asChild
              size="sm"
              className="h-8 text-xs bg-primary hover:bg-primary/90 text-white font-medium"
            >
              <Link to="/analyze">
                Analyze Transaction <ArrowRight className="size-3 ml-1" />
              </Link>
            </Button>
          </div>
        </div>
      )}

      {/* Summary KPI Cards using MetricCard */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          title="Total Transactions"
          value={isBackendError && !showDemoFallback ? "—" : total}
          subtitle={kpiNote("Persisted audit events")}
          icon={Activity}
          tone="primary"
          accentColor="border-l-primary"
          isLoading={statsLoading}
        />
        <MetricCard
          title="Legitimate Volume"
          value={isBackendError && !showDemoFallback ? "—" : legitimate}
          subtitle={kpiNote("Passed security baseline")}
          icon={ShieldCheck}
          tone="success"
          accentColor="border-l-emerald-500"
          isLoading={statsLoading}
        />
        <MetricCard
          title="Fraudulent Flagged"
          value={isBackendError && !showDemoFallback ? "—" : flagged}
          subtitle={kpiNote("Model flagged fraud risk")}
          icon={AlertTriangle}
          tone="danger"
          accentColor="border-l-red-500"
          isLoading={statsLoading}
        />
        <MetricCard
          title="Audit Fraud Rate"
          value={isBackendError && !showDemoFallback ? "—" : `${fraudRate}%`}
          subtitle={kpiNote("Incidence among processed")}
          icon={Percent}
          tone={fraudRate > 15 ? "danger" : "warning"}
          accentColor={fraudRate > 15 ? "border-l-red-500" : "border-l-amber-500"}
          isLoading={statsLoading}
        />
      </div>

      {/* Charts Grid */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Trend Chart */}
        <Card className="lg:col-span-2 border-border/80 bg-card shadow-card flex flex-col justify-between">
          <div>
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div>
                <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
                  <TrendingUp className="size-4 text-primary" />
                  Transaction Volume & Flagged Patterns
                </CardTitle>
                <CardDescription className="text-xs text-muted-foreground mt-0.5">
                  Daily evaluation throughput vs. model-flagged fraudulent transactions
                </CardDescription>
              </div>
              <Badge
                variant="outline"
                className="text-[10px] font-mono border-border bg-elevated/60 text-muted-foreground"
              >
                {useLive && hasLiveTransactions
                  ? `Rolling 30D: ${trendReportingPeriod}`
                  : "Demo Trend"}
              </Badge>
            </CardHeader>
            <CardContent className="h-72 pt-2">
              {trendData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trendData} margin={{ left: -16, right: 8, top: 8, bottom: 0 }}>
                    <defs>
                      <linearGradient id="txnFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.4} />
                        <stop offset="90%" stopColor="#3B82F6" stopOpacity={0.0} />
                      </linearGradient>
                      <linearGradient id="fraudFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#EF4444" stopOpacity={0.3} />
                        <stop offset="90%" stopColor="#EF4444" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                    <XAxis
                      dataKey="day"
                      tickLine={false}
                      axisLine={false}
                      fontSize={11}
                      stroke="#64748B"
                    />
                    <YAxis tickLine={false} axisLine={false} fontSize={11} stroke="#64748B" />
                    <Tooltip
                      contentStyle={{
                        borderRadius: 8,
                        border: "1px solid #253147",
                        background: "#111827",
                        color: "#F1F5F9",
                        fontSize: 12,
                        boxShadow: "0 10px 25px -5px rgba(0,0,0,0.5)",
                      }}
                      labelStyle={{ color: "#94A3B8", fontWeight: 600, marginBottom: 4 }}
                    />
                    <Area
                      type="monotone"
                      dataKey="transactions"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      fill="url(#txnFill)"
                      name="Total Volume"
                    />
                    <Area
                      type="monotone"
                      dataKey="flagged"
                      stroke="#EF4444"
                      strokeWidth={2}
                      fill="url(#fraudFill)"
                      name="Flagged Fraud"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState
                  icon={Activity}
                  title="No Trend Data Recorded"
                  description="Analyzed transactions will be plotted chronologically as volume accumulates."
                  className="py-12 border-0 bg-transparent"
                />
              )}
            </CardContent>
          </div>
          {useLive && hasLiveTransactions && (
            <div className="px-5 py-2.5 text-[11px] text-muted-foreground border-t border-border/40 bg-elevated/20 rounded-b-lg flex flex-col sm:flex-row sm:items-center sm:justify-between gap-1">
              <span>
                Rolling 30-Day Window:{" "}
                <strong className="text-foreground font-mono">{trendReportingPeriod}</strong> (
                {trendData.reduce((s: number, i) => s + (i.transactions || 0), 0)} transactions
                across {trendData.length} active dates)
              </span>
              <div className="flex items-center gap-2 text-slate-400 italic">
                {historicalExcludedCount > 0 && (
                  <span>
                    {historicalExcludedCount} historical record(s) prior to {windowStartDate}{" "}
                    preserved in full audit log.
                  </span>
                )}
                {futureExcludedCount > 0 && (
                  <span>({futureExcludedCount} future record(s) excluded)</span>
                )}
              </div>
            </div>
          )}
        </Card>

        {/* Prediction Distribution Pie Chart */}
        <Card className="border-border/80 bg-card shadow-card">
          <CardHeader className="flex flex-row items-center justify-between pb-3">
            <div>
              <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
                <PieIcon className="size-4 text-emerald-400" />
                Classification Split
              </CardTitle>
              <CardDescription className="text-xs text-muted-foreground mt-0.5">
                Ratio of legitimate to flagged transactions
              </CardDescription>
            </div>
            <Badge
              variant="outline"
              className="text-[10px] font-mono border-border bg-elevated/60 text-muted-foreground"
            >
              {useLive && hasLiveTransactions ? "Live Ratio" : "Demo Ratio"}
            </Badge>
          </CardHeader>
          <CardContent className="h-72 pt-2">
            {total > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={distributionData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={62}
                    outerRadius={92}
                    paddingAngle={4}
                    stroke="#090D16"
                    strokeWidth={2}
                  >
                    {distributionData.map((entry) => (
                      <Cell
                        key={entry.name}
                        fill={entry.name.toLowerCase().includes("fraud") ? "#EF4444" : "#10B981"}
                      />
                    ))}
                  </Pie>
                  <Legend
                    verticalAlign="bottom"
                    iconType="circle"
                    iconSize={8}
                    wrapperStyle={{ fontSize: 11, paddingTop: 12, color: "#94A3B8" }}
                  />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 8,
                      border: "1px solid #253147",
                      background: "#111827",
                      color: "#F1F5F9",
                      fontSize: 12,
                      boxShadow: "0 10px 25px -5px rgba(0,0,0,0.5)",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <EmptyState
                icon={PieIcon}
                title="No Predictions Available"
                description="Transaction classifications will render when data is evaluated."
                className="py-12 border-0 bg-transparent"
              />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Recent Transactions Table */}
      <Card className="border-border/80 bg-card shadow-card">
        <CardHeader className="flex flex-row items-center justify-between pb-4">
          <div>
            <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
              <Clock className="size-4 text-primary" />
              Recent Audit Log
            </CardTitle>
            <CardDescription className="text-xs text-muted-foreground mt-0.5">
              {useLive && hasLiveTransactions
                ? "Recent transactions audited by the active Indian UPI Random Forest engine."
                : showDemoFallback || !apiConfigured
                  ? "Synthetic transactions displayed for demonstration purposes."
                  : isBackendError
                    ? "Audit records unreachable from backend."
                    : "No transactions recorded yet."}
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            asChild
            className="text-xs text-primary hover:text-primary/90 hover:bg-primary/10"
          >
            <Link to="/history">
              Full History <ArrowRight className="size-3.5 ml-1" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent className="p-0">
          {txLoading ? (
            <LoadingState
              message="Loading recent transactions from audit database…"
              className="py-12"
            />
          ) : (txError || isBackendError) && !showDemoFallback ? (
            <EmptyState
              icon={AlertTriangle}
              title="Backend Unreachable"
              description="Unable to retrieve audit history from the Flask backend service."
              action={{
                label: "Retry Query",
                onClick: () => refetchTx(),
              }}
              className="py-12"
            />
          ) : useLive && !hasLiveTransactions && !showDemoFallback ? (
            <EmptyState
              icon={Database}
              title="Audit Log is Empty"
              description="No transaction events have been processed yet. Submit your first transaction for machine learning analysis."
              action={{
                label: "Analyze First Transaction",
                href: "/analyze",
              }}
              className="py-12"
            />
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader className="bg-elevated/40">
                  <TableRow className="border-b border-border/80 hover:bg-transparent">
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider pl-6">
                      Txn ID
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Amount
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Channel
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Timestamp
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      ML Verdict
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider pr-6">
                      Risk Band
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {displayRows.map((t) => {
                    const formatted = formatTransactionAmount(t.amount, t.currency);
                    return (
                      <TableRow
                        key={t.id}
                        className="border-b border-border/40 hover:bg-elevated/30 transition-colors"
                      >
                        <TableCell className="font-mono text-xs text-slate-300 pl-6">
                          {t.id}
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono font-medium text-xs text-foreground">
                              {formatted.text}
                            </span>
                            {formatted.isLegacy && (
                              <Badge
                                variant="outline"
                                className="text-[9px] font-normal border-border/60 text-muted-foreground bg-elevated/40 px-1 py-0"
                              >
                                Legacy
                              </Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <span className="inline-flex items-center rounded px-2 py-0.5 text-xs font-mono font-medium bg-elevated/80 border border-border/60 text-slate-300">
                            {t.channel}
                          </span>
                        </TableCell>
                        <TableCell className="text-xs">
                          {t.hasTransactionTime ? (
                            <div className="space-y-0.5">
                              <span className="font-medium text-foreground block">
                                {formatTimestamp(t.timestamp)}
                              </span>
                              {t.createdAt && (
                                <span className="text-[10px] text-muted-foreground font-mono">
                                  Logged: {formatTimestamp(t.createdAt)}
                                </span>
                              )}
                            </div>
                          ) : (
                            <span className="text-slate-300">{formatTimestamp(t.timestamp)}</span>
                          )}
                        </TableCell>
                        <TableCell>
                          <StatusBadge label={t.label} />
                        </TableCell>
                        <TableCell className="pr-6">
                          <RiskBadge
                            level={
                              t.riskLevel ||
                              (t.label.toLowerCase().includes("fraud") ||
                              t.label.toLowerCase().includes("flagged")
                                ? "High"
                                : "Low")
                            }
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
