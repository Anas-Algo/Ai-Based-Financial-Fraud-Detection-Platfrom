import * as React from "react";
import { AlertTriangle } from "lucide-react";

export function PageHeader({
  title,
  description,
  badge,
  children,
}: {
  title: string;
  description: string;
  badge?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <header className="flex flex-col gap-3 pb-2 border-b border-border/40 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <div className="flex items-center gap-2.5">
          <h1 className="truncate text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
            {title}
          </h1>
          {badge}
        </div>
        <p className="mt-1 text-xs text-muted-foreground sm:text-sm">{description}</p>
      </div>
      {children ? <div className="flex items-center gap-2 shrink-0">{children}</div> : null}
    </header>
  );
}

export function DemoBanner({ text }: { text: string }) {
  return (
    <div
      role="note"
      className="flex items-center gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 text-xs font-medium text-amber-300 shadow-sm"
    >
      <AlertTriangle className="size-4 shrink-0 text-amber-400" aria-hidden />
      <span>{text}</span>
    </div>
  );
}
