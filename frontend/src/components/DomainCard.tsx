import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface DomainCardProps {
  number: number;
  title: string;
  description: string;
  examples: string;
  icon: React.ReactNode;
  delay?: number;
}

export function DomainCard({ number, title, description, examples, icon, delay = 0 }: DomainCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div
      className={cn(
        "group bg-card rounded-2xl border border-border/50 shadow-card hover:shadow-elevated transition-all duration-300 overflow-hidden opacity-0 animate-fade-up",
        isExpanded && "ring-2 ring-primary/20"
      )}
      style={{ animationDelay: `${delay}ms`, animationFillMode: "forwards" }}
    >
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full text-left p-6 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-inset"
      >
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-secondary flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors duration-300">
            {icon}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Domain {number}
              </span>
            </div>
            <h3 className="font-serif text-xl font-semibold text-foreground group-hover:text-primary transition-colors">
              {title}
            </h3>
            <p className="mt-2 text-muted-foreground text-sm leading-relaxed line-clamp-2">
              {description}
            </p>
          </div>
          <ChevronDown
            className={cn(
              "flex-shrink-0 w-5 h-5 text-muted-foreground transition-transform duration-300",
              isExpanded && "rotate-180"
            )}
          />
        </div>
      </button>

      <div
        className={cn(
          "grid transition-all duration-300 ease-out",
          isExpanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
        )}
      >
        <div className="overflow-hidden">
          <div className="px-6 pb-6 pt-2 border-t border-border/50">
            <p className="text-sm text-muted-foreground leading-relaxed">
              <span className="font-medium text-foreground">Examples: </span>
              {examples}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
