import { Link, useRouterState } from "@tanstack/react-router";
import {
  LayoutDashboard,
  ScanSearch,
  History,
  BarChart3,
  BookOpen,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
  Database,
  Terminal,
} from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const mainNavItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, exact: true },
  { to: "/analyze", label: "Analyze Transaction", icon: ScanSearch, exact: false },
  { to: "/history", label: "Transaction History", icon: History, exact: false },
  { to: "/analytics", label: "Analytics & Risk", icon: BarChart3, exact: false },
] as const;

const secondaryNavItems = [
  { to: "/about", label: "Methodology", icon: BookOpen, exact: false },
] as const;

export function SidebarBrand({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <Link to="/" className="flex items-center gap-3 px-4 py-5 group transition-colors">
      <div className="relative grid size-10 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-primary/90 to-primary/60 text-white shadow-md shadow-primary/20 ring-1 ring-primary/40">
        <ShieldCheck className="size-5" aria-hidden />
        <span className="absolute -top-0.5 -right-0.5 size-2 rounded-full bg-cyan-400 ring-2 ring-sidebar" />
      </div>
      {!collapsed ? (
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="truncate text-base font-bold tracking-tight text-slate-100">
              FraudLens
            </span>
            <span className="rounded bg-primary/20 px-1 py-0.2 text-[9px] font-semibold text-primary uppercase">
              AI
            </span>
          </div>
          <p className="truncate text-[10px] uppercase tracking-wider text-slate-400 font-medium">
            Academic Risk Prototype
          </p>
        </div>
      ) : null}
    </Link>
  );
}

export function SidebarNav({
  onNavigate,
  collapsed = false,
}: {
  onNavigate?: () => void;
  collapsed?: boolean;
}) {
  const routerState = useRouterState();
  const currentPath = routerState.location.pathname;

  return (
    <div className="flex flex-1 flex-col justify-between py-2">
      <div className="space-y-4 px-3">
        <div>
          {!collapsed ? (
            <p className="px-3 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Operations
            </p>
          ) : null}
          <nav className="flex flex-col gap-1" aria-label="Main">
            {mainNavItems.map(({ to, label, icon: Icon, exact }) => {
              const isActive = exact ? currentPath === to : currentPath.startsWith(to);
              return (
                <Link
                  key={to}
                  to={to}
                  onClick={onNavigate}
                  activeOptions={{ exact }}
                  title={collapsed ? label : undefined}
                  className={cn(
                    "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-xs font-medium transition-all duration-150",
                    isActive
                      ? "bg-primary/15 text-primary font-semibold shadow-inner"
                      : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200",
                    collapsed && "justify-center px-2",
                  )}
                >
                  {isActive ? (
                    <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r-full bg-primary" />
                  ) : null}
                  <Icon
                    className={cn(
                      "size-4 shrink-0 transition-colors",
                      isActive ? "text-primary" : "text-slate-400 group-hover:text-slate-200",
                    )}
                    aria-hidden
                  />
                  {!collapsed ? <span className="truncate">{label}</span> : null}
                </Link>
              );
            })}
          </nav>
        </div>

        <div>
          {!collapsed ? (
            <p className="px-3 pb-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              System
            </p>
          ) : null}
          <nav className="flex flex-col gap-1" aria-label="System">
            {secondaryNavItems.map(({ to, label, icon: Icon, exact }) => {
              const isActive = exact ? currentPath === to : currentPath.startsWith(to);
              return (
                <Link
                  key={to}
                  to={to}
                  onClick={onNavigate}
                  activeOptions={{ exact }}
                  title={collapsed ? label : undefined}
                  className={cn(
                    "group relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-xs font-medium transition-all duration-150",
                    isActive
                      ? "bg-primary/15 text-primary font-semibold shadow-inner"
                      : "text-slate-400 hover:bg-slate-800/60 hover:text-slate-200",
                    collapsed && "justify-center px-2",
                  )}
                >
                  {isActive ? (
                    <span className="absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r-full bg-primary" />
                  ) : null}
                  <Icon
                    className={cn(
                      "size-4 shrink-0 transition-colors",
                      isActive ? "text-primary" : "text-slate-400 group-hover:text-slate-200",
                    )}
                    aria-hidden
                  />
                  {!collapsed ? <span className="truncate">{label}</span> : null}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Sidebar Footer Info Card */}
      {!collapsed ? (
        <div className="p-3">
          <div className="rounded-xl border border-border/80 bg-slate-900/90 p-3 shadow-sm">
            <div className="flex items-center justify-between pb-1.5">
              <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                Inference Engine
              </span>
              <span className="flex items-center gap-1 text-[10px] font-mono text-emerald-400">
                <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Active
              </span>
            </div>
            <p className="text-[11px] font-medium text-slate-300">Indian UPI Model (INR)</p>
            <p className="text-[10px] text-slate-500 mt-0.5">80-tree Random Forest</p>
          </div>
        </div>
      ) : null}
    </div>
  );
}

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "hidden shrink-0 flex-col border-r border-sidebar-border bg-sidebar transition-all duration-200 md:flex",
        collapsed ? "w-16" : "w-64",
      )}
    >
      <div className="flex items-center justify-between border-b border-sidebar-border/60">
        <SidebarBrand collapsed={collapsed} />
        {!collapsed ? (
          <button
            type="button"
            onClick={() => setCollapsed(true)}
            title="Collapse sidebar"
            className="mr-3 grid size-7 place-items-center rounded-md border border-border bg-card/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <ChevronLeft className="size-4" />
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setCollapsed(false)}
            title="Expand sidebar"
            className="mx-auto my-3 grid size-7 place-items-center rounded-md border border-border bg-card/60 text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <ChevronRight className="size-4" />
          </button>
        )}
      </div>

      <SidebarNav collapsed={collapsed} />
    </aside>
  );
}
