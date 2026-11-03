import * as React from "react";
import { cn } from "@/lib/utils";

export interface RiskBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  risk?: string;
  level?: string;
}

export function RiskBadge({ risk, level, className, ...props }: RiskBadgeProps) {
  const displayRisk = risk || level || "Low";
  const normRisk = displayRisk.toLowerCase();
  const isHigh = normRisk.includes("high");
  const isMed = normRisk.includes("medium");

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium border uppercase tracking-wider",
        isHigh
          ? "border-destructive/30 bg-destructive/10 text-red-400"
          : isMed
            ? "border-amber-500/30 bg-amber-500/10 text-amber-400"
            : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
        className,
      )}
      {...props}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          isHigh ? "bg-red-400" : isMed ? "bg-amber-400" : "bg-emerald-400",
        )}
        aria-hidden
      />
      <span>{displayRisk}</span>
    </div>
  );
}
