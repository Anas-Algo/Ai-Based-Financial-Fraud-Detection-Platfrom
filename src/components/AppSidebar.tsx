import { Link } from "@tanstack/react-router";
import { LayoutDashboard, ScanSearch, History, BarChart3, Info, ShieldCheck } from "lucide-react";

const navItems = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, exact: true },
  { to: "/analyze", label: "Analyze Transaction", icon: ScanSearch },
  { to: "/history", label: "Transaction History", icon: History },
  { to: "/analytics", label: "Analytics", icon: BarChart3 },
  { to: "/about", label: "About", icon: Info },
] as const;

export function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex flex-col gap-1 p-3" aria-label="Main">
      {navItems.map(({ to, label, icon: Icon, ...rest }) => (
        <Link
          key={to}
          to={to}
          onClick={onNavigate}
          activeOptions={{ exact: "exact" in rest ? rest.exact : false }}
          className="group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-sidebar-foreground/70 transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground data-[status=active]:bg-sidebar-primary data-[status=active]:text-sidebar-primary-foreground"
        >
          <Icon className="size-4 shrink-0" aria-hidden />
          <span className="truncate">{label}</span>
        </Link>
      ))}
    </nav>
  );
}

export function SidebarBrand() {
  return (
    <div className="flex items-center gap-3 px-5 py-5">
      <span className="grid size-9 shrink-0 place-items-center rounded-xl bg-sidebar-primary text-sidebar-primary-foreground">
        <ShieldCheck className="size-5" aria-hidden />
      </span>
      <div className="min-w-0">
        <p className="truncate text-sm font-semibold text-sidebar-foreground">FraudLens</p>
        <p className="truncate text-xs text-sidebar-foreground/60">Academic prototype</p>
      </div>
    </div>
  );
}

export function AppSidebar() {
  return (
    <aside className="hidden w-64 shrink-0 flex-col border-r border-sidebar-border bg-sidebar md:flex">
      <SidebarBrand />
      <SidebarNav />
      <div className="mt-auto p-4">
        <p className="rounded-lg bg-sidebar-accent p-3 text-xs leading-relaxed text-sidebar-foreground/70">
          Educational prototype. Synthetic data only — not a real banking system.
        </p>
      </div>
    </aside>
  );
}
