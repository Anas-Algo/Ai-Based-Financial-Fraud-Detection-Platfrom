import { Link, useRouterState } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  ArrowRight,
  Database,
  ExternalLink,
  RefreshCw,
  ScanSearch,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { healthQueryOptions } from "@/lib/api-queries";
import { isApiConfigured } from "@/lib/fraud-api";

export function AppHeader({ onToggleMobileNav }: { onToggleMobileNav?: () => void }) {
  const routerState = useRouterState();
  const currentPath = routerState.location.pathname;

  const apiConfigured = isApiConfigured();
  const { data: health, isError, isFetching, refetch } = useQuery(healthQueryOptions());

  const getPageTitle = (path: string) => {
    switch (path) {
      case "/":
        return "Command Center";
      case "/analyze":
        return "Transaction Analysis";
      case "/history":
        return "Audit History";
      case "/analytics":
        return "Risk Intelligence";
      case "/about":
        return "System Methodology";
      default:
        return "Platform";
    }
  };

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center justify-between border-b border-border/80 bg-background/80 px-4 sm:px-8 backdrop-blur-md">
      {/* Left: Route Breadcrumb & Title */}
      <div className="flex items-center gap-3">
        {onToggleMobileNav ? (
          <button
            type="button"
            onClick={onToggleMobileNav}
            className="grid size-9 place-items-center rounded-lg border border-border bg-card text-foreground md:hidden"
            aria-label="Toggle navigation"
          >
            <Activity className="size-4" />
          </button>
        ) : null}
        <div className="hidden sm:flex items-center gap-2 text-xs text-muted-foreground">
          <span className="font-semibold text-slate-300">FraudLens</span>
          <span>/</span>
          <span className="text-foreground font-medium">{getPageTitle(currentPath)}</span>
        </div>
        <Badge
          variant="outline"
          className="hidden md:inline-flex border-border/60 bg-elevated/60 text-[10px] text-muted-foreground font-mono"
        >
          Model: Indian UPI (80-Trees)
        </Badge>
      </div>

      {/* Right: API Health Status, Quick Action & Refresh */}
      <div className="flex items-center gap-2.5">
        {/* Backend health status badge */}
        {apiConfigured ? (
          <div
            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium border ${
              health?.status === "healthy"
                ? "border-emerald-500/25 bg-emerald-500/10 text-emerald-400"
                : isError
                  ? "border-destructive/30 bg-destructive/10 text-red-400"
                  : "border-border bg-card text-muted-foreground"
            }`}
          >
            <span
              className={`size-1.5 rounded-full ${
                health?.status === "healthy"
                  ? "bg-emerald-400 animate-pulse"
                  : isError
                    ? "bg-red-400"
                    : "bg-amber-400 animate-ping"
              }`}
            />
            <span className="hidden sm:inline">
              {health?.status === "healthy"
                ? "ML Engine Live"
                : isError
                  ? "API Offline"
                  : "Connecting…"}
            </span>
          </div>
        ) : (
          <Badge variant="outline" className="text-[10px] text-amber-400 border-amber-500/30">
            Demo Mode
          </Badge>
        )}

        {/* Quick action button to Analyze if on other pages */}
        {currentPath !== "/analyze" ? (
          <Button
            asChild
            size="sm"
            className="h-8 px-3 text-xs bg-primary hover:bg-primary/90 text-white shadow-sm font-medium"
          >
            <Link to="/analyze">
              <ScanSearch className="size-3.5 mr-1.5" />
              <span className="hidden sm:inline">Analyze Txn</span>
              <span className="sm:hidden">Analyze</span>
            </Link>
          </Button>
        ) : null}
      </div>
    </header>
  );
}
