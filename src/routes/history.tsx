import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { Search } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
import { DEMO_NOTICE, demoTransactions, formatAmount, formatTimestamp } from "@/lib/demo-data";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "Transaction History — FraudLens Prototype" },
      {
        name: "description",
        content:
          "Searchable and filterable table of synthetic sample transactions used by the academic fraud detection prototype.",
      },
      { property: "og:title", content: "Transaction History — FraudLens Prototype" },
      {
        property: "og:description",
        content: "Search and filter the synthetic sample transaction records.",
      },
    ],
  }),
  component: HistoryPage,
});

const filters = ["All", "Legitimate (sample)", "Flagged (sample)"] as const;

function HistoryPage() {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<(typeof filters)[number]>("All");

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return demoTransactions.filter((t) => {
      const matchesQuery =
        !q || t.id.toLowerCase().includes(q) || t.channel.toLowerCase().includes(q);
      const matchesFilter = filter === "All" || t.label === filter;
      return matchesQuery && matchesFilter;
    });
  }, [query, filter]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
      <PageHeader
        title="Transaction History"
        description="Browse the synthetic sample transaction records."
      />
      <DemoBanner text={DEMO_NOTICE} />

      <Card className="shadow-card">
        <CardContent className="flex flex-col gap-4 p-5">
          <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
            <div className="grid min-w-0 gap-2">
              <Label htmlFor="search">Search by ID or channel</Label>
              <div className="relative">
                <Search
                  className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden
                />
                <Input
                  id="search"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="TXN-100245"
                  className="bg-card pl-9"
                />
              </div>
            </div>
            <div className="grid gap-2">
              <Label htmlFor="status">Sample label</Label>
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

          <p className="text-sm text-muted-foreground">
            Showing {rows.length} of {demoTransactions.length} sample records.
          </p>

          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Transaction ID</TableHead>
                  <TableHead>Sample amount</TableHead>
                  <TableHead>Timestamp (UTC)</TableHead>
                  <TableHead>Prediction</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((t) => (
                  <TableRow key={t.id}>
                    <TableCell className="font-mono text-xs">{t.id}</TableCell>
                    <TableCell>{formatAmount(t.amount)}</TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatTimestamp(t.timestamp)}
                    </TableCell>
                    <TableCell className="text-muted-foreground">Not available</TableCell>
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
                {rows.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="py-10 text-center text-muted-foreground">
                      No sample records match your search.
                    </TableCell>
                  </TableRow>
                ) : null}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
