// SYNTHETIC DEMONSTRATION DATA ONLY.
// None of this comes from a model, database, or real transactions.
// It exists purely so the academic prototype UI has something to render.

export type DemoTransaction = {
  id: string;
  amount: number;
  timestamp: string;
  channel: "Card" | "Transfer" | "Wallet" | "ATM";
  label: "Legitimate (sample)" | "Flagged (sample)";
};

export const DEMO_NOTICE =
  "Synthetic demonstration data — no trained model, API, or database is connected.";

const channels: DemoTransaction["channel"][] = ["Card", "Transfer", "Wallet", "ATM"];

function pseudoRandom(seed: number) {
  const x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

export const demoTransactions: DemoTransaction[] = Array.from({ length: 48 }, (_, i) => {
  const r = pseudoRandom(i + 1);
  const flagged = r > 0.88;
  const date = new Date(Date.UTC(2026, 8, 1 + Math.floor(i / 2), 9 + (i % 12), (i * 7) % 60));
  return {
    id: `TXN-${(100245 + i * 13).toString()}`,
    amount: Math.round((50 + r * 9450) * 100) / 100,
    timestamp: date.toISOString(),
    channel: channels[i % channels.length]!,
    label: flagged ? "Flagged (sample)" : "Legitimate (sample)",
  };
});

export const demoTotals = {
  total: demoTransactions.length,
  legitimate: demoTransactions.filter((t) => t.label === "Legitimate (sample)").length,
  flagged: demoTransactions.filter((t) => t.label === "Flagged (sample)").length,
};

export const demoFraudRate = Math.round((demoTotals.flagged / demoTotals.total) * 1000) / 10;

export const demoTrend = Array.from({ length: 12 }, (_, i) => {
  const r = pseudoRandom(i + 20);
  return {
    day: `Day ${i + 1}`,
    transactions: 120 + Math.round(r * 80),
    flagged: 4 + Math.round(r * 9),
  };
});

export const demoDistribution = [
  { name: "Legitimate (sample)", value: demoTotals.legitimate },
  { name: "Flagged (sample)", value: demoTotals.flagged },
];

export const demoByChannel = channels.map((c) => ({
  channel: c,
  legitimate: demoTransactions.filter((t) => t.channel === c && t.label !== "Flagged (sample)")
    .length,
  flagged: demoTransactions.filter((t) => t.channel === c && t.label === "Flagged (sample)").length,
}));

export const demoAmountBands = [
  { band: "0–1k", count: demoTransactions.filter((t) => t.amount < 1000).length },
  {
    band: "1k–3k",
    count: demoTransactions.filter((t) => t.amount >= 1000 && t.amount < 3000).length,
  },
  {
    band: "3k–6k",
    count: demoTransactions.filter((t) => t.amount >= 3000 && t.amount < 6000).length,
  },
  { band: "6k+", count: demoTransactions.filter((t) => t.amount >= 6000).length },
];

export function formatAmount(value: number) {
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR" }).format(value);
}

export function formatTimestamp(iso: string) {
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
}
