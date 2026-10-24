import { createFileRoute, Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertCircle, Database, History, Loader2, RefreshCw, Search } from "lucide-react";

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
import { transactionsQueryOptions } from "@/lib/api-queries";
import { formatAmount, formatTimestamp } from "@/lib/demo-data";
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

function HistoryPage() {
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
        title="Transaction History"
        description="Searchable audit trail of transactions processed by the AI fraud detection pipeline."
      >
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
            className="h-8 text-xs"
          >
            <RefreshCw className={`size-3.5 mr-1 ${isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </div>
      </PageHeader>

      {!apiConfigured && (
        <div className="rounded-xl border border-warning-border bg-warning-surface p-4 text-sm text-warning-foreground">
          ML backend URL is not configured. Please set{" "}
          <code className="font-mono">VITE_FLASK_API_URL</code> in your environment.
        </div>
      )}

      {isError && (
        <div className="flex items-center justify-between rounded-xl border border-destructive/20 bg-destructive/10 p-4 text-sm text-destructive">
          <div className="flex items-center gap-2">
            <AlertCircle className="size-4 shrink-0" />
            <span>
              Failed to load transaction audit log:{" "}
              {error instanceof Error ? error.message : "Unknown error"}
            </span>
          </div>
          <Button variant="outline" size="sm" onClick={() => refetch()} className="text-xs">
            Retry
          </Button>
        </div>
      )}

      <Card className="shadow-card">
        <CardContent className="flex flex-col gap-4 p-5">
          <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
            <div className="grid min-w-0 gap-2">
              <Label htmlFor="search">Search by ID, channel, or merchant</Label>
              <div className="relative">
                <Search
                  className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden
                />
                <Input
                  id="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="e.g. TXN-100245 or Card"
                  className="bg-card pl-9"
                />
              </div>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="status">Filter by Classification</Label>
              <select
                id="status"
                value={filter}
                onChange={(e) => setFilter(e.target.value as (typeof filters)[number])}
                className="h-10 rounded-md border border-input bg-card px-3 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {filters.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <p>
              Showing {rows.length} of {transactions.length} audit records.
            </p>
            {isFetching && (
              <span className="flex items-center gap-1 text-primary">
                <Loader2 className="size-3 animate-spin" /> Updating…
              </span>
            )}
          </div>

          <div className="overflow-x-auto">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center gap-2 py-16 text-muted-foreground">
                <Loader2 className="size-8 animate-spin text-primary" />
                <p className="text-sm">Loading audit history from SQLite database…</p>
              </div>
            ) : transactions.length === 0 ? (
              <div className="flex flex-col items-center justify-center gap-3 py-16 text-center text-muted-foreground">
                <Database className="size-10 opacity-30" />
                <p className="text-base font-semibold text-foreground">
                  No Transactions Analyzed Yet
                </p>
                <p className="max-w-md text-xs">
                  The SQLite database is currently empty. Go to the Analyze page to submit a
                  transaction and see its classification and audit log recorded here.
                </p>
                <Button asChild size="sm" className="mt-2">
                  <Link to="/analyze">Analyze First Transaction</Link>
                </Button>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Transaction ID</TableHead>
                    <TableHead>Amount</TableHead>
                    <TableHead>Channel</TableHead>
                    <TableHead>Category</TableHead>
                    <TableHead>Transaction Time</TableHead>
                    <TableHead>Model Prediction</TableHead>
                    <TableHead>Fraud Probability</TableHead>
                    <TableHead>Risk Level</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((t) => (
                    <TableRow key={t.transactionId}>
                      <TableCell className="font-mono text-xs font-medium">
                        {t.transactionId}
                      </TableCell>
                      <TableCell className="font-medium">{formatAmount(t.amount)}</TableCell>
                      <TableCell className="text-xs">{t.channel}</TableCell>
                      <TableCell className="text-xs capitalize">{t.merchantCategory}</TableCell>
                      <TableCell className="text-xs">
                        <div
                          className={
                            t.hasTransactionTime ? "font-medium text-foreground" : "text-foreground"
                          }
                        >
                          {formatTimestamp(t.timestamp)}
                        </div>
                        {t.hasTransactionTime && t.createdAt ? (
                          <div className="text-[10px] text-muted-foreground">
                            Analyzed: {formatTimestamp(t.createdAt)}
                          </div>
                        ) : (
                          <div className="text-[10px] text-muted-foreground/75 italic">
                            Recorded at analysis
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant="outline"
                          className={
                            t.label === "Fraudulent" || t.isFraud === 1
                              ? "border-destructive/30 bg-destructive/10 text-destructive font-semibold"
                              : "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 font-semibold"
                          }
                        >
                          {t.label}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs font-mono">
                        {t.fraudProbability != null
                          ? `${(t.fraudProbability * 100).toFixed(1)}%`
                          : "N/A"}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            t.riskLevel === "High"
                              ? "destructive"
                              : t.riskLevel === "Medium"
                                ? "outline"
                                : "default"
                          }
                          className="text-[10px] uppercase font-bold"
                        >
                          {t.riskLevel || "Low"}
                        </Badge>
                      </TableCell>
                    </TableRow>
                  ))}
                  {rows.length === 0 && transactions.length > 0 && (
                    <TableRow>
                      <TableCell
                        colSpan={8}
                        className="py-10 text-center text-muted-foreground text-sm"
                      >
                        No transactions match your current search and filter criteria.
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
