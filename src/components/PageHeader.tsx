export function PageHeader({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children?: React.ReactNode;
}) {
  return (
    <header className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-4 sm:flex sm:flex-wrap sm:items-center sm:justify-between">
      <div className="min-w-0">
        <h1 className="truncate text-2xl font-semibold tracking-tight text-foreground">{title}</h1>
        <p className="mt-1 text-sm text-muted-foreground">{description}</p>
      </div>
      {children}
    </header>
  );
}

export function DemoBanner({ text }: { text: string }) {
  return (
    <div
      role="note"
      className="rounded-xl border border-warning-border bg-warning-surface px-4 py-3 text-sm text-warning-foreground"
    >
      {text}
    </div>
  );
}
