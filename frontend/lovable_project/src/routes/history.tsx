import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertCircle,
  Database,
  History,
  Loader2,
  RefreshCw,
  Search,
  ShieldCheck,
} from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import { PageHeader } from "@/components/PageHeader";
import { StatusBadge } from "@/components/StatusBadge";
import { RiskBadge } from "@/components/RiskBadge";
import { EmptyState } from "@/components/EmptyState";
import { LoadingState } from "@/components/LoadingState";
import { transactionsQueryOptions } from "@/lib/api-queries";
import { formatTimestamp, formatTransactionAmount } from "@/lib/demo-data";
import { isApiConfigured, type ApiTransaction } from "@/lib/fraud-api";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Transaction History — FraudLens AI Audit Log" },
      {
        name: "description",
        content:
          "Searchable and filterable audit log of evaluated financial transactions from the FraudLens SQLite database.",
      },
      { property: "og:title", content: "Transaction History — FraudLens AI Audit Log" },
      {
        property: "og:description",
        content:
          "Audit log of evaluated financial transactions with model confidence and risk classifications.",
      },
    ],
  }),
  component: HistoryPage,
});

const filters = ["All", "Legitimate", "Fraudulent"] as const;

export function HistoryPage() {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<(typeof filters)[number]>("All");

  const apiConfigured = isApiConfigured();
  const { data, isLoading, isError, error, refetch, isFetching } = useQuery(
    transactionsQueryOptions(200),
  );

  const transactions: ApiTransaction[] = useMemo(
    () => data?.transactions ?? [],
    [data?.transactions],
  );

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return transactions.filter((t) => {
      const idMatch = t.transactionId.toLowerCase().includes(q);
      const channelMatch = t.channel.toLowerCase().includes(q);
      const catMatch = (t.merchantCategory || "").toLowerCase().includes(q);
      const matchesQuery = !q || idMatch || channelMatch || catMatch;

      const matchesFilter =
        filter === "All" ||
        (filter === "Fraudulent" && (t.label === "Fraudulent" || t.isFraud === 1)) ||
        (filter === "Legitimate" && (t.label === "Legitimate" || t.isFraud === 0));

      return matchesQuery && matchesFilter;
    });
  }, [transactions, query, filter]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
      <PageHeader
        title="Audit History & Logs"
        description="Comprehensive immutable audit record of evaluated transactions, confidence metrics, and feature derivations."
      >
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
            className="h-8 text-xs border-border bg-card hover:bg-elevated text-slate-300"
          >
            <RefreshCw className={`size-3.5 mr-1.5 ${isFetching ? "animate-spin" : ""}`} />
            Refresh Log
          </Button>
        </div>
      </PageHeader>

      {!apiConfigured && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-3.5 text-xs sm:text-sm text-amber-300">
          ML backend URL is not configured. Set{" "}
          <code className="font-mono bg-black/40 px-1 py-0.5 rounded text-amber-200">
            VITE_FLASK_API_URL
          </code>{" "}
          in your environment.
        </div>
      )}

      {isError && (
        <div className="flex items-center justify-between rounded-lg border border-red-500/30 bg-red-950/20 p-4 text-sm text-red-300">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="size-4 shrink-0 text-red-400" />
            <span>
              Failed to load transaction audit repository:{" "}
              {error instanceof Error ? error.message : "Unknown error"}
            </span>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            className="h-8 text-xs border-red-500/40 bg-red-950/40 hover:bg-red-900/30 text-red-200"
          >
            Retry
          </Button>
        </div>
      )}

      <Card className="border-border/80 bg-card shadow-card">
        <CardContent className="flex flex-col gap-4 p-5">
          {/* Controls Bar */}
          <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_220px] sm:items-end">
            <div className="grid min-w-0 gap-1.5">
              <Label htmlFor="search" className="text-xs font-medium text-slate-300">
                Filter by ID, channel, or merchant category
              </Label>
              <div className="relative">
                <Search
                  className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground"
                  aria-hidden
                />
                <Input
                  id="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. TXN-100245, UPI, or Kirana"
                  className="bg-elevated/50 border-border text-foreground text-xs pl-8 placeholder:text-muted-foreground focus-visible:ring-primary"
                />
              </div>
            </div>
            <div className="grid gap-1.5">
              <Label htmlFor="status" className="text-xs font-medium text-slate-300">
                Filter by Verdict
              </Label>
              <select
                id="status"
                value={filter}
                onChange={(e) => setFilter(e.target.value as (typeof filters)[number])}
                className="h-9 rounded-md border border-border bg-elevated/50 px-3 text-xs text-foreground outline-none focus:border-primary focus:ring-1 focus:ring-primary"
              >
                {filters.map((f) => (
                  <option key={f} value={f} className="bg-card text-foreground">
                    {f} Transactions
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Counts pill & loading indicator */}
          <div className="flex items-center justify-between text-xs text-muted-foreground pt-1 border-t border-border/40">
            <span className="font-mono text-[11px]">
              Showing <strong className="text-foreground">{rows.length}</strong> of{" "}
              <strong className="text-foreground">{transactions.length}</strong> audited events
            </span>
            {isFetching && (
              <span className="flex items-center gap-1.5 text-primary text-[11px] font-medium">
                <Loader2 className="size-3 animate-spin" /> Syncing database…
              </span>
            )}
          </div>

          {/* Table */}
          <div className="overflow-x-auto rounded-md border border-border/60">
            {isLoading ? (
              <LoadingState
                message="Querying transaction audit records from SQLite…"
                className="py-16"
              />
            ) : transactions.length === 0 ? (
              <EmptyState
                icon={Database}
                title="Audit Database is Empty"
                description="No transactions have been evaluated yet. Run an analysis to generate the first permanent audit record."
                action={{
                  label: "Analyze First Transaction",
                  href: "/analyze",
                }}
                className="py-16"
              />
            ) : (
              <Table>
                <TableHeader className="bg-elevated/40">
                  <TableRow className="border-b border-border/80 hover:bg-transparent">
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider pl-4">
                      Transaction ID
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Amount
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Channel
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Category
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Timestamp
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Verdict
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Fraud Prob.
                    </TableHead>
                    <TableHead className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider pr-4">
                      Risk Band
                    </TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((t) => (
                    <TableRow
                      key={t.transactionId}
                      className="border-b border-border/40 hover:bg-elevated/30 transition-colors"
                    >
                      <TableCell className="font-mono text-xs text-slate-300 font-medium pl-4">
                        {t.transactionId}
                      </TableCell>
                      <TableCell className="font-medium">
                        {(() => {
                          const formatted = formatTransactionAmount(t.amount, t.currency);
                          return (
                            <div className="flex items-center gap-1.5">
                              <span className="font-mono text-xs text-foreground font-semibold">
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
                          );
                        })()}
                      </TableCell>
                      <TableCell>
                        <span className="inline-flex items-center rounded px-2 py-0.5 text-xs font-mono font-medium bg-elevated/80 border border-border/60 text-slate-300">
                          {t.channel}
                        </span>
                      </TableCell>
                      <TableCell className="text-xs text-slate-300">
                        {t.merchantCategory || "—"}
                      </TableCell>
                      <TableCell className="text-xs">
                        {t.hasTransactionTime ? (
                          <div className="space-y-0.5">
                            <span className="font-medium text-foreground block">
                              {formatTimestamp(t.timestamp)}
                            </span>
                            {t.createdAt && (
                              <span className="text-[10px] text-muted-foreground font-mono block">
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
                      <TableCell className="text-xs font-mono font-medium text-slate-300">
                        {t.fraudProbability != null
                          ? `${(t.fraudProbability * 100).toFixed(1)}%`
                          : "N/A"}
                      </TableCell>
                      <TableCell className="pr-4">
                        <RiskBadge
                          level={
                            t.riskLevel ||
                            (t.label === "Fraudulent" || t.isFraud === 1 ? "High" : "Low")
                          }
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                  {rows.length === 0 && transactions.length > 0 && (
                    <TableRow>
                      <TableCell
                        colSpan={8}
                        className="py-12 text-center text-muted-foreground text-xs"
                      >
                        No audit events match your search and filter criteria.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
