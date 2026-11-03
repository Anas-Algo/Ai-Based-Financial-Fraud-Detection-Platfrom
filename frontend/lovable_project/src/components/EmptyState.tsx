import * as React from "react";
import { type LucideIcon } from "lucide-react";
import { Link } from "@tanstack/react-router";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface EmptyStateAction {
  label: string;
  onClick?: () => void;
  href?: string;
}

interface EmptyStateProps {
  icon: LucideIcon;
  title: string;
  description: string;
  action?: React.ReactNode | EmptyStateAction;
  className?: string;
}

export function EmptyState({ icon: Icon, title, description, action, className }: EmptyStateProps) {
  const renderAction = () => {
    if (!action) return null;
    if (React.isValidElement(action)) return action;

    const actionObj = action as EmptyStateAction;
    if (actionObj.href) {
      return (
        <Button
          asChild
          size="sm"
          className="h-8 text-xs bg-primary hover:bg-primary/90 text-white font-medium"
        >
          <Link to={actionObj.href}>{actionObj.label}</Link>
        </Button>
      );
    }
    if (actionObj.onClick) {
      return (
        <Button
          size="sm"
          variant="outline"
          onClick={actionObj.onClick}
          className="h-8 text-xs border-border bg-card hover:bg-elevated text-slate-300"
        >
          {actionObj.label}
        </Button>
      );
    }
    return null;
  };

  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 py-12 px-4 text-center",
        className,
      )}
    >
      <div className="grid size-12 place-items-center rounded-2xl border border-border bg-elevated text-muted-foreground shadow-sm">
        <Icon className="size-6 opacity-70 text-slate-400" aria-hidden />
      </div>
      <div className="max-w-sm space-y-1">
        <h3 className="text-sm font-semibold text-foreground tracking-tight">{title}</h3>
        <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
      </div>
      {action ? <div className="mt-2">{renderAction()}</div> : null}
    </div>
  );
}
