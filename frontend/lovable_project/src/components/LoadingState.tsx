import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface LoadingStateProps {
  message?: string;
  className?: string;
}

export function LoadingState({
  message = "Loading real-time records…",
  className,
}: LoadingStateProps) {
  return (
    <div
      className={cn("flex flex-col items-center justify-center gap-3 py-14 text-center", className)}
    >
      <div className="relative">
        <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
        <div className="absolute inset-0 size-8 animate-ping rounded-full bg-primary/20" />
      </div>
      <p className="text-xs font-medium text-muted-foreground tracking-wide">{message}</p>
    </div>
  );
}
