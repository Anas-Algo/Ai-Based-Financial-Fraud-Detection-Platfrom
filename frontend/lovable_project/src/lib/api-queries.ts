// React Query options for REAL Flask API data.
import { queryOptions } from "@tanstack/react-query";

import { checkHealth, getAnalytics, getStats, getTransactions, isApiConfigured } from "./fraud-api";

export const apiKeys = {
  all: ["flask-api"] as const,
  stats: () => [...apiKeys.all, "stats"] as const,
  transactions: (limit?: number) => [...apiKeys.all, "transactions", limit ?? 100] as const,
  analytics: () => [...apiKeys.all, "analytics"] as const,
  health: () => [...apiKeys.all, "health"] as const,
};

const common = {
  enabled: isApiConfigured(),
  retry: 1,
  staleTime: 10_000,
  refetchOnWindowFocus: false,
} as const;

export const statsQueryOptions = () =>
  queryOptions({ queryKey: apiKeys.stats(), queryFn: getStats, ...common });

export const transactionsQueryOptions = (limit = 100) =>
  queryOptions({
    queryKey: apiKeys.transactions(limit),
    queryFn: () => getTransactions(limit),
    ...common,
  });

export const analyticsQueryOptions = () =>
  queryOptions({ queryKey: apiKeys.analytics(), queryFn: getAnalytics, ...common });

export const healthQueryOptions = () =>
  queryOptions({ queryKey: apiKeys.health(), queryFn: checkHealth, ...common });

export type DataSource = "demo" | "api";
