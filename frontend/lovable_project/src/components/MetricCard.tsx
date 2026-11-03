import * as React from "react";
import { type LucideIcon, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface MetricCardProps {
  label?: string;
  title?: string;
  value: React.ReactNode;
  icon: LucideIcon;
  note?: string;
  subtitle?: string;
  tone?: "primary" | "success" | "danger" | "warning" | "cyan";
  loading?: boolean;
  isLoading?: boolean;
  accentColor?: string;
  className?: string;
}

export function MetricCard({
  label,
  title,
  value,
  icon: Icon,
  note,
  subtitle,
  tone = "primary",
  loading,
  isLoading,
  accentColor,
  className,
}: MetricCardProps) {
  const displayLabel = label || title || "";
  const displayNote = note || subtitle;
  const isCardLoading = Boolean(loading ?? isLoading ?? false);

  const toneMap = {
    primary: {
      iconBg: "bg-primary/10 text-primary border-primary/20",
      accent: "text-primary",
      borderAccent: "border-l-primary",
    },
    success: {
      iconBg: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
      accent: "text-emerald-400",
      borderAccent: "border-l-emerald-500",
    },
    danger: {
      iconBg: "bg-destructive/10 text-red-400 border-destructive/20",
      accent: "text-red-400",
      borderAccent: "border-l-red-500",
    },
    warning: {
      iconBg: "bg-amber-500/10 text-amber-400 border-amber-500/20",
      accent: "text-amber-400",
      borderAccent: "border-l-amber-500",
    },
    cyan: {
      iconBg: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
      accent: "text-cyan-400",
      borderAccent: "border-l-cyan-400",
    },
  };

  const currentTone = toneMap[tone] || toneMap.primary;

  return (
    <Card
      className={cn(
        "relative overflow-hidden border-border/80 bg-card shadow-card transition-all hover:border-border",
        accentColor || `border-l-2 ${currentTone.borderAccent}`,
        className,
      )}
    >
      <CardContent className="flex items-center justify-between gap-4 p-5">
        <div className="min-w-0 flex-1">
          <p className="truncate text-xs font-medium uppercase tracking-wider text-muted-foreground">
            {displayLabel}
          </p>
          <div className="mt-2 flex items-baseline gap-2">
            {isCardLoading ? (
              <Loader2 className="size-6 animate-spin text-muted-foreground" aria-hidden />
            ) : (
              <span className="font-mono text-2xl font-bold tracking-tight text-foreground">
                {value}
              </span>
            )}
          </div>
          {displayNote ? (
            <p className="mt-1 truncate text-xs text-muted-foreground">{displayNote}</p>
          ) : null}
        </div>
        <div
          className={cn(
            "grid size-11 shrink-0 place-items-center rounded-xl border",
            currentTone.iconBg,
          )}
        >
          <Icon className="size-5" aria-hidden />
        </div>
      </CardContent>
    </Card>
  );
}
