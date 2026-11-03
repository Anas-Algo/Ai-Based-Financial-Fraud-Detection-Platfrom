import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Database,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  PieChart as PieIcon,
  TrendingUp,
  Layers,
  ArrowRight,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PageHeader, DemoBanner } from "@/components/PageHeader";
import { EmptyState } from "@/components/EmptyState";
import { LoadingState } from "@/components/LoadingState";
import { analyticsQueryOptions, statsQueryOptions } from "@/lib/api-queries";
import { isApiConfigured } from "@/lib/fraud-api";
import { demoAmountBands, demoByChannel, demoDistribution, demoTrend } from "@/lib/demo-data";

export const Route = createFileRoute("/analytics")({
  head: () => ({
    meta: [
      { title: "Risk Intelligence & Analytics — FraudLens" },
      {
        name: "description",
        content:
          "AI fraud analytics and visual breakdown of transaction classifications by channel, amount band, and volume trend.",
      },
      { property: "og:title", content: "Risk Intelligence & Analytics — FraudLens" },
      {
        property: "og:description",
        content:
          "Breakdown of financial transactions by prediction label, channel distribution, and amount bands.",
      },
    ],
  }),
  component: AnalyticsPage,
});

const darkTooltipStyle = {
  borderRadius: 8,
  border: "1px solid #253147",
  background: "#111827",
  color: "#F1F5F9",
  fontSize: 12,
  boxShadow: "0 10px 25px -5px rgba(0,0,0,0.5)",
};

export function AnalyticsPage() {
  const [showSamplePreview, setShowSamplePreview] = useState(false);

  const apiConfigured = isApiConfigured();
  const {
    data: analytics,
    isLoading: anLoading,
    isError: anError,
    error: anErrorObj,
    refetch: refetchAnalytics,
    isFetching: anFetching,
  } = useQuery(analyticsQueryOptions());
  const {
    data: stats,
    isLoading: statsLoading,
    isError: statsError,
    refetch: refetchStats,
    isFetching: statsFetching,
  } = useQuery(statsQueryOptions());

  const isError = anError || statsError;
  const isLoading = anLoading || statsLoading;
  const isFetching = anFetching || statsFetching;

  const handleRefresh = () => {
    refetchAnalytics();
    refetchStats();
  };

  const totalTransactions = analytics?.totalTransactions ?? stats?.totalTransactions ?? 0;
  const hasLiveTransactions = totalTransactions > 0;
  const isLiveMode = apiConfigured && !isError && !showSamplePreview;

  const getBadgeText = (liveText: string, sampleText: string) => {
    if (showSamplePreview || !apiConfigured) return sampleText;
    if (isError) return "Unavailable";
    return liveText;
  };

  // Channels chart data with grouping of unsupported/legacy channels under "Other / Legacy"
  const supportedChannels = useMemo(
    () => ["UPI", "Card", "Transfer", "Wallet", "ATM"] as const,
    [],
  );

  const { channelData, legacyChannelCount } = useMemo(() => {
    if (!isLiveMode) {
      return {
        channelData: showSamplePreview || !apiConfigured ? demoByChannel : [],
        legacyChannelCount: 0,
      };
    }

    const raw = analytics?.channelBreakdown ?? [];
    const map = new Map<string, { channel: string; legitimate: number; flagged: number }>();
    for (const ch of supportedChannels) {
      map.set(ch, { channel: ch, legitimate: 0, flagged: 0 });
    }

    let otherLegit = 0;
    let otherFlagged = 0;
    let hasOther = false;

    for (const item of raw) {
      if (map.has(item.channel)) {
        const entry = map.get(item.channel)!;
        entry.legitimate += item.legitimate;
        entry.flagged += item.flagged;
      } else {
        hasOther = true;
        otherLegit += item.legitimate;
        otherFlagged += item.flagged;
      }
    }

    const result = Array.from(map.values());
    if (hasOther) {
      result.push({
        channel: "Other / Legacy",
        legitimate: otherLegit,
        flagged: otherFlagged,
      });
    }

    return {
      channelData: result,
      legacyChannelCount: otherLegit + otherFlagged,
    };
  }, [
    isLiveMode,
    analytics?.channelBreakdown,
    showSamplePreview,
    apiConfigured,
    supportedChannels,
  ]);

  // Distribution chart data
  const distData = isLiveMode
    ? analytics?.labelDistribution && analytics.labelDistribution.some((d) => d.value > 0)
      ? analytics.labelDistribution
      : []
    : showSamplePreview || !apiConfigured
      ? demoDistribution
      : [];

  // Amount bands data
  const amountData = isLiveMode
    ? (analytics?.amountBands ?? [])
    : showSamplePreview || !apiConfigured
      ? demoAmountBands
      : [];

  // Trend data
  const trendData = isLiveMode
    ? (analytics?.transactionTrend ?? [])
    : showSamplePreview || !apiConfigured
      ? demoTrend
      : [];

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
      <PageHeader
        title="Risk Intelligence & Analytics"
        description="Aggregated risk intelligence, channel distribution, and transaction denomination metrics from simulated evaluation sessions."
      >
        <div className="flex items-center gap-2">
          {apiConfigured && (
            <Badge
              variant="outline"
              className={
                isLiveMode && hasLiveTransactions
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400 font-medium"
                  : "border-border bg-card text-muted-foreground"
              }
            >
              <Database className="size-3 mr-1 text-emerald-400" />
              {isLiveMode && hasLiveTransactions ? "Live Aggregates" : "Engine Connected"}
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

      {!apiConfigured ? (
        <DemoBanner text="ML Backend is not configured. Displaying synthetic preview analytics." />
      ) : isError ? (
        <div className="flex items-center justify-between rounded-lg border border-red-500/30 bg-red-950/20 p-4 text-sm text-red-300">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="size-4 shrink-0 text-red-400" />
            <span>
              Could not fetch analytics from Flask backend:{" "}
              {anErrorObj instanceof Error
                ? anErrorObj.message
                : "Backend unavailable. Verify that the Flask service is running."}
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            className="h-8 text-xs border-red-500/40 bg-red-950/40 hover:bg-red-900/30 text-red-200"
          >
            Retry Connection
          </Button>
        </div>
      ) : hasLiveTransactions ? (
        <div className="flex items-center justify-between rounded-lg border border-emerald-500/25 bg-emerald-500/5 px-4 py-3 text-sm text-emerald-300">
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="size-4 shrink-0 text-emerald-400" />
            <span className="text-xs sm:text-sm">
              Visualizing aggregates from{" "}
              <strong className="font-mono text-emerald-300">{totalTransactions}</strong> evaluated
              transactions in local SQLite audit log.
            </span>
          </div>
        </div>
      ) : (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-lg border border-primary/30 bg-primary/5 px-4 py-3 text-sm text-slate-300">
          <div className="flex items-center gap-2.5">
            <Sparkles className="size-4 text-primary shrink-0" />
            <span className="text-xs sm:text-sm">
              Audit log currently has 0 transactions evaluated.
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSamplePreview(!showSamplePreview)}
              className="h-8 text-xs border-border bg-card hover:bg-elevated text-slate-300"
            >
              {showSamplePreview ? "Hide Preview" : "View Synthetic Preview"}
            </Button>
            <Button
              asChild
              size="sm"
              className="h-8 text-xs bg-primary hover:bg-primary/90 text-white font-medium"
            >
              <Link to="/analyze">
                Analyze a Transaction <ArrowRight className="size-3 ml-1" />
              </Link>
            </Button>
          </div>
        </div>
      )}

      {isLoading ? (
        <LoadingState message="Aggregating metrics across transaction log…" className="py-20" />
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Channel Breakdown */}
          <Card className="border-border/80 bg-card shadow-card">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div>
                <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
                  <BarChart3 className="size-4 text-primary" />
                  Volume by Payment Rail
                </CardTitle>
                <CardDescription className="text-xs text-muted-foreground mt-0.5">
                  Legitimate vs. flagged transactions across payment rails
                </CardDescription>
              </div>
              <Badge
                variant="outline"
                className="text-[10px] font-mono border-border bg-elevated/60 text-muted-foreground"
              >
                {getBadgeText("Live Data", "Demo Data")}
              </Badge>
            </CardHeader>
            <CardContent className="h-72 pt-2">
              {channelData.length > 0 &&
              channelData.some((c) => c.legitimate > 0 || c.flagged > 0) ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={channelData} margin={{ left: -16, right: 8, top: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                    <XAxis
                      dataKey="channel"
                      tickLine={false}
                      axisLine={false}
                      fontSize={11}
                      stroke="#64748B"
                    />
                    <YAxis tickLine={false} axisLine={false} fontSize={11} stroke="#64748B" />
                    <Tooltip contentStyle={darkTooltipStyle} />
                    <Legend
                      iconType="circle"
                      iconSize={8}
                      wrapperStyle={{ fontSize: 11, color: "#94A3B8" }}
                    />
                    <Bar
                      dataKey="legitimate"
                      name="Legitimate"
                      fill="#10B981"
                      radius={[4, 4, 0, 0]}
                    />
                    <Bar
                      dataKey="flagged"
                      name="Fraud Flagged"
                      fill="#EF4444"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState
                  icon={BarChart3}
                  title="No Rail Activity Recorded"
                  description="Activity grouped by channel will plot here when transactions are processed."
                  className="py-12 border-0 bg-transparent"
                />
              )}
            </CardContent>
            {legacyChannelCount > 0 && isLiveMode && (
              <div className="px-5 py-2.5 text-[11px] text-muted-foreground border-t border-border/40 bg-elevated/20 rounded-b-lg">
                <span className="italic text-slate-400">
                  {legacyChannelCount} legacy audit record (such as web) grouped under Other /
                  Legacy to ensure full audit reconciliation.
                </span>
              </div>
            )}
          </Card>

          {/* Label Split */}
          <Card className="border-border/80 bg-card shadow-card">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div>
                <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
                  <PieIcon className="size-4 text-emerald-400" />
                  Aggregate Ratio
                </CardTitle>
                <CardDescription className="text-xs text-muted-foreground mt-0.5">
                  Proportion of audited transactions flagged vs approved
                </CardDescription>
              </div>
              <Badge
                variant="outline"
                className="text-[10px] font-mono border-border bg-elevated/60 text-muted-foreground"
              >
                {getBadgeText("Live Ratio", "Demo Ratio")}
              </Badge>
            </CardHeader>
            <CardContent className="h-72 pt-2">
              {distData.length > 0 && distData.some((d) => d.value > 0) ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={distData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={62}
                      outerRadius={92}
                      paddingAngle={4}
                      stroke="#090D16"
                      strokeWidth={2}
                    >
                      {distData.map((entry) => (
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
                    <Tooltip contentStyle={darkTooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState
                  icon={ShieldCheck}
                  title="No Classifications Recorded"
                  description="Overall class distribution ratio will appear here once transactions are audited."
                  className="py-12 border-0 bg-transparent"
                />
              )}
            </CardContent>
          </Card>

          {/* Amount Bands */}
          <Card className="border-border/80 bg-card shadow-card">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div>
                <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
                  <Layers className="size-4 text-cyan-400" />
                  Amount Brackets ({analytics?.amountBandsCurrency || "INR"})
                </CardTitle>
                <CardDescription className="text-xs text-muted-foreground mt-0.5">
                  Transaction count distribution by transaction value brackets
                </CardDescription>
              </div>
              <Badge
                variant="outline"
                className="text-[10px] font-mono border-border bg-elevated/60 text-muted-foreground"
              >
                {getBadgeText(
                  `${analytics?.amountBandsCurrency || "INR"} Denominated`,
                  "Demo Bands",
                )}
              </Badge>
            </CardHeader>
            <CardContent className="space-y-3 pt-2">
              <div className="h-64">
                {amountData.length > 0 && amountData.some((b) => b.count > 0) ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={amountData} margin={{ left: -16, right: 8, top: 8, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                      <XAxis
                        dataKey="band"
                        tickLine={false}
                        axisLine={false}
                        fontSize={11}
                        stroke="#64748B"
                      />
                      <YAxis tickLine={false} axisLine={false} fontSize={11} stroke="#64748B" />
                      <Tooltip contentStyle={darkTooltipStyle} />
                      <Bar
                        dataKey="count"
                        name={`${analytics?.amountBandsCurrency || "INR"} Transactions`}
                        fill="#3B82F6"
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <EmptyState
                    icon={Activity}
                    title={`No ${analytics?.amountBandsCurrency || "INR"} Transactions Yet`}
                    description={
                      isLiveMode && (analytics?.unspecifiedAmountTotal ?? 0) > 0
                        ? `The audit database has ${analytics?.unspecifiedAmountTotal} historical record(s) with segregated currencies to avoid cross-currency skew.`
                        : `New ${analytics?.amountBandsCurrency || "INR"} transactions will be categorized by value bracket here.`
                    }
                    className="py-12 border-0 bg-transparent"
                  />
                )}
              </div>
              {isLiveMode && (analytics?.unspecifiedAmountTotal ?? 0) > 0 && (
                <div className="rounded-md border border-border/60 bg-elevated/40 p-2.5 text-[11px] text-muted-foreground leading-normal">
                  <span className="font-semibold text-slate-300">Currency Disclosure:</span>{" "}
                  {analytics?.amountBandsDisclosure ||
                    "Amount distribution aggregates transactions denominated in Indian Rupees (INR). Historical records in other currencies are segregated."}{" "}
                  ({analytics?.unspecifiedAmountTotal} record
                  {(analytics?.unspecifiedAmountTotal ?? 0) > 1 ? "s" : ""} segregated).
                </div>
              )}
            </CardContent>
          </Card>

          {/* Volume Over Time */}
          <Card className="border-border/80 bg-card shadow-card">
            <CardHeader className="flex flex-row items-center justify-between pb-3">
              <div>
                <CardTitle className="text-sm font-semibold tracking-tight text-foreground flex items-center gap-2">
                  <TrendingUp className="size-4 text-red-400" />
                  Fraud Incidence Timeline
                </CardTitle>
                <CardDescription className="text-xs text-muted-foreground mt-0.5">
                  Daily tracking of anomalous and fraudulent attempts
                </CardDescription>
              </div>
              <Badge
                variant="outline"
                className="text-[10px] font-mono border-border bg-elevated/60 text-muted-foreground"
              >
                {getBadgeText("Live Timeline", "Demo Trend")}
              </Badge>
            </CardHeader>
            <CardContent className="h-72 pt-2">
              {trendData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData} margin={{ left: -16, right: 8, top: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
                    <XAxis
                      dataKey="day"
                      tickLine={false}
                      axisLine={false}
                      fontSize={11}
                      stroke="#64748B"
                    />
                    <YAxis tickLine={false} axisLine={false} fontSize={11} stroke="#64748B" />
                    <Tooltip contentStyle={darkTooltipStyle} />
                    <Line
                      type="monotone"
                      dataKey="flagged"
                      name="Fraud Flagged"
                      stroke="#EF4444"
                      strokeWidth={2}
                      dot={{ r: 3, fill: "#EF4444", strokeWidth: 0 }}
                      activeDot={{ r: 5, strokeWidth: 0 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <EmptyState
                  icon={Activity}
                  title="No Timeline Data Recorded"
                  description="Daily time-series telemetry will populate as transaction audit history accumulates."
                  className="py-12 border-0 bg-transparent"
                />
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
