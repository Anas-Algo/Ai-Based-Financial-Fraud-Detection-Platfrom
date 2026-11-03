import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
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
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PageHeader, DemoBanner } from "@/components/PageHeader";
import { analyticsQueryOptions, statsQueryOptions } from "@/lib/api-queries";
import { isApiConfigured } from "@/lib/fraud-api";
import { demoAmountBands, demoByChannel, demoDistribution, demoTrend } from "@/lib/demo-data";

export const Route = createFileRoute("/analytics")({
  head: () => ({
    meta: [
      { title: "Analytics — FraudLens AI Analytics" },
      {
        name: "description",
        content:
          "AI fraud analytics and visual breakdown of transaction classifications by channel, amount band, and volume trend.",
      },
      { property: "og:title", content: "Analytics — FraudLens AI Analytics" },
      {
        property: "og:description",
        content:
          "Breakdown of financial transactions by prediction label, channel distribution, and amount bands.",
      },
    ],
  }),
  component: AnalyticsPage,
});

const tooltipStyle = {
  borderRadius: 12,
  border: "1px solid var(--border)",
  background: "var(--card)",
};

const pieColors = ["#10b981", "#ef4444"];

function AnalyticsPage() {
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

  // Channels chart data
  const channelData = isLiveMode
    ? (analytics?.channelBreakdown ?? [])
    : showSamplePreview || !apiConfigured
      ? demoByChannel
      : [];

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
        title="Analytics & Distribution"
        description="Detailed visual analysis of transaction activity, fraud channels, and risk distributions."
      >
        <div className="flex items-center gap-2">
          {apiConfigured && (
            <Badge
              variant="outline"
              className={
                isLiveMode && hasLiveTransactions
                  ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
                  : "border-muted"
              }
            >
              <Database className="size-3 mr-1" />
              {isLiveMode && hasLiveTransactions ? "Live Database Charts" : "Backend Connected"}
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

      {!apiConfigured ? (
        <DemoBanner text="ML Backend is not configured. Displaying synthetic preview analytics." />
      ) : isError ? (
        <div className="flex items-center justify-between rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
          <div className="flex items-center gap-2">
            <AlertCircle className="size-4 shrink-0" />
            <span>
              Could not fetch analytics from Flask backend:{" "}
              {anErrorObj instanceof Error
                ? anErrorObj.message
                : "Backend unavailable. Please check that the Flask service is running."}
            </span>
          </div>
          <Button variant="outline" size="sm" onClick={handleRefresh} className="text-xs">
            Retry Connection
          </Button>
        </div>
      ) : hasLiveTransactions ? (
        <div className="flex items-center justify-between rounded-xl border border-emerald-500/20 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-800 dark:text-emerald-200">
          <div className="flex items-center gap-2">
            <ShieldCheck className="size-4 shrink-0 text-emerald-600" />
            <span>Visualizing live aggregates from {totalTransactions} analyzed transactions.</span>
          </div>
        </div>
      ) : (
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 rounded-xl border border-primary/20 bg-primary/5 px-4 py-3 text-sm">
          <div className="flex items-center gap-2">
            <Sparkles className="size-4 text-primary shrink-0" />
            <span>The database has 0 transactions analyzed so far.</span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowSamplePreview(!showSamplePreview)}
              className="h-8 text-xs"
            >
              {showSamplePreview ? "Hide Sample Preview" : "View Sample Preview"}
            </Button>
            <Button asChild size="sm" className="h-8 text-xs">
              <Link to="/analyze">Analyze a Transaction</Link>
            </Button>
          </div>
        </div>
      )}

      {isLoading ? (
        <div className="flex flex-col items-center justify-center gap-3 py-20 text-muted-foreground">
          <Loader2 className="size-8 animate-spin text-primary" />
          <p className="text-sm">Calculating aggregated analytics from SQLite database…</p>
        </div>
      ) : (
        <div className="grid gap-4 lg:grid-cols-2">
          {/* Channel Breakdown */}
          <Card className="shadow-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">Transactions by Payment Channel</CardTitle>
              <Badge variant="outline" className="text-xs font-normal">
                {getBadgeText("Live Data", "Sample Data")}
              </Badge>
            </CardHeader>
            <CardContent className="h-72">
              {channelData.length > 0 &&
              channelData.some((c) => c.legitimate > 0 || c.flagged > 0) ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={channelData} margin={{ left: -16, right: 8, top: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="channel" tickLine={false} axisLine={false} fontSize={12} />
                    <YAxis tickLine={false} axisLine={false} fontSize={12} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend iconType="circle" />
                    <Bar
                      dataKey="legitimate"
                      name="Legitimate"
                      fill="#10b981"
                      radius={[6, 6, 0, 0]}
                    />
                    <Bar
                      dataKey="flagged"
                      name="Fraud Flagged"
                      fill="#ef4444"
                      radius={[6, 6, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
                  <BarChart3 className="size-8 opacity-40" />
                  <p className="text-sm">No channel activity recorded yet.</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Label Split */}
          <Card className="shadow-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">Class Distribution Ratio</CardTitle>
              <Badge variant="outline" className="text-xs font-normal">
                {getBadgeText("Live Ratio", "Sample Ratio")}
              </Badge>
            </CardHeader>
            <CardContent className="h-72">
              {distData.length > 0 && distData.some((d) => d.value > 0) ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={distData} dataKey="value" nameKey="name" outerRadius={95}>
                      {distData.map((entry) => (
                        <Cell
                          key={entry.name}
                          fill={entry.name.toLowerCase().includes("fraud") ? "#ef4444" : "#10b981"}
                        />
                      ))}
                    </Pie>
                    <Legend verticalAlign="bottom" iconType="circle" />
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
                  <ShieldCheck className="size-8 opacity-40" />
                  <p className="text-sm">No classified transactions yet.</p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Amount Bands */}
          <Card className="shadow-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">
                Transaction Amount Distribution ({analytics?.amountBandsCurrency || "INR"})
              </CardTitle>
              <Badge variant="outline" className="text-xs font-normal">
                {getBadgeText(
                  `${analytics?.amountBandsCurrency || "INR"} Denominated`,
                  "Sample Bands",
                )}
              </Badge>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="h-64">
                {amountData.length > 0 && amountData.some((b) => b.count > 0) ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={amountData} margin={{ left: -16, right: 8, top: 8 }}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="var(--border)"
                        vertical={false}
                      />
                      <XAxis dataKey="band" tickLine={false} axisLine={false} fontSize={12} />
                      <YAxis tickLine={false} axisLine={false} fontSize={12} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Bar
                        dataKey="count"
                        name={`${analytics?.amountBandsCurrency || "INR"} Transactions`}
                        fill="var(--chart-5)"
                        radius={[6, 6, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center text-muted-foreground">
                    <Activity className="size-8 opacity-40" />
                    <p className="text-sm font-medium">
                      No {analytics?.amountBandsCurrency || "INR"} transactions recorded yet.
                    </p>
                    {isLiveMode && (analytics?.unspecifiedAmountTotal ?? 0) > 0 ? (
                      <p className="max-w-xs text-xs">
                        The database contains {analytics?.unspecifiedAmountTotal} historical record
                        {analytics?.unspecifiedAmountTotal === 1 ? "" : "s"} with segregated
                        currencies, preserved to prevent cross-currency distortion.
                      </p>
                    ) : (
                      <p className="text-xs">
                        New {analytics?.amountBandsCurrency || "INR"} transactions will be plotted
                        here.
                      </p>
                    )}
                  </div>
                )}
              </div>
              {isLiveMode && (analytics?.unspecifiedAmountTotal ?? 0) > 0 ? (
                <div className="rounded-lg border border-border/60 bg-muted/20 p-2.5 text-[11px] text-muted-foreground leading-normal">
                  <span className="font-semibold text-foreground">Currency Disclosure:</span>{" "}
                  {analytics?.amountBandsDisclosure ||
                    "Amount distribution aggregates transactions denominated in Indian Rupees (INR). Historical records in other currencies are segregated."}{" "}
                  ({analytics?.unspecifiedAmountTotal} record
                  {(analytics?.unspecifiedAmountTotal ?? 0) > 1 ? "s" : ""} segregated).
                </div>
              ) : null}
            </CardContent>
          </Card>

          {/* Volume Over Time */}
          <Card className="shadow-card">
            <CardHeader className="flex flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">Fraud Volume Timeline</CardTitle>
              <Badge variant="outline" className="text-xs font-normal">
                {getBadgeText("Live Timeline", "Sample Trend")}
              </Badge>
            </CardHeader>
            <CardContent className="h-72">
              {trendData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData} margin={{ left: -16, right: 8, top: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                    <XAxis dataKey="day" tickLine={false} axisLine={false} fontSize={12} />
                    <YAxis tickLine={false} axisLine={false} fontSize={12} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line
                      type="monotone"
                      dataKey="flagged"
                      name="Fraudulent"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={true}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-2 text-center text-muted-foreground">
                  <Activity className="size-8 opacity-40" />
                  <p className="text-sm">No daily timeline data recorded yet.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
