import * as React from "react";
import { AlertOctagon, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface StatusBadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  status?: string;
  label?: string;
  showIcon?: boolean;
}

export function StatusBadge({
  status,
  label,
  showIcon = true,
  className,
  ...props
}: StatusBadgeProps) {
  const displayStatus = status || label || "Legitimate";
  const isFraud =
    displayStatus === "Fraudulent" ||
    displayStatus === "Flagged (sample)" ||
    displayStatus.toLowerCase().includes("fraud") ||
    displayStatus.toLowerCase().includes("flagged");

  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold tracking-wide border transition-colors",
        isFraud
          ? "border-destructive/30 bg-destructive/10 text-red-400"
          : "border-emerald-500/30 bg-emerald-500/10 text-emerald-400",
        className,
      )}
      {...props}
    >
      {showIcon ? (
        isFraud ? (
          <AlertOctagon className="size-3 shrink-0 text-red-400" aria-hidden />
        ) : (
          <CheckCircle2 className="size-3 shrink-0 text-emerald-400" aria-hidden />
        )
      ) : (
        <span
          className={cn("size-1.5 rounded-full", isFraud ? "bg-red-400" : "bg-emerald-400")}
          aria-hidden
        />
      )}
      <span>{displayStatus}</span>
    </div>
  );
}
