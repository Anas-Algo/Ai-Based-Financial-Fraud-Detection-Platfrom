// React Query options for REAL Flask API data.
// Queries are disabled until VITE_FLASK_API_URL is set, so nothing is fetched
// while the backend does not exist. Pages keep rendering demo-data.ts today;
// later, swap a page to `useQuery(statsQueryOptions())` and drop the demo import.
import { queryOptions } from "@tanstack/react-query";

import { getAnalytics, getStats, getTransactions, isApiConfigured } from "./fraud-api";

export const apiKeys = {
  all: ["flask-api"] as const,
  stats: () => [...apiKeys.all, "stats"] as const,
  transactions: () => [...apiKeys.all, "transactions"] as const,
  analytics: () => [...apiKeys.all, "analytics"] as const,
};

const common = { enabled: isApiConfigured(), retry: 1, staleTime: 30_000 } as const;

export const statsQueryOptions = () =>
  queryOptions({ queryKey: apiKeys.stats(), queryFn: getStats, ...common });

export const transactionsQueryOptions = () =>
  queryOptions({ queryKey: apiKeys.transactions(), queryFn: getTransactions, ...common });

export const analyticsQueryOptions = () =>
  queryOptions({ queryKey: apiKeys.analytics(), queryFn: getAnalytics, ...common });

/** Which source a page is showing. Today always "demo" until the API is wired in. */
export type DataSource = "demo" | "api";
