import { Check, Target } from "lucide-react";
import {
  MATCH_ACCEPTANCE_TRADEOFFS,
  MATCH_TRADEOFFS,
  type MatchAcceptanceKey,
  type MatchTradeoffKey,
} from "@/lib/matching";

interface MatchTradeoffControlsProps {
  selected: MatchTradeoffKey[];
  onChange: (selected: MatchTradeoffKey[]) => void;
  accepted?: MatchAcceptanceKey[];
  onAcceptedChange?: (selected: MatchAcceptanceKey[]) => void;
}

export function MatchTradeoffControls({
  selected,
  onChange,
  accepted = [],
  onAcceptedChange,
}: MatchTradeoffControlsProps) {
  const toggle = (key: MatchTradeoffKey) => {
    const next = selected.includes(key)
      ? selected.filter((item) => item !== key)
      : [...selected, key];
    onChange(next);

    if (onAcceptedChange) {
      const validAcceptances = MATCH_ACCEPTANCE_TRADEOFFS.filter((item) =>
        item.requires.some((required) => next.includes(required))
      ).map((item) => item.key);
      onAcceptedChange(accepted.filter((item) => validAcceptances.includes(item)));
    }
  };

  const toggleAccepted = (key: MatchAcceptanceKey) => {
    if (!onAcceptedChange) return;
    onAcceptedChange(
      accepted.includes(key) ? accepted.filter((item) => item !== key) : [...accepted, key]
    );
  };

  const availableAcceptances = MATCH_ACCEPTANCE_TRADEOFFS.filter((item) =>
    item.requires.some((required) => selected.includes(required))
  );

  return (
    <div className="grid gap-4">
      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
          Goals
        </p>
        <div className="flex flex-wrap gap-2">
          {MATCH_TRADEOFFS.map((tradeoff) => {
            const active = selected.includes(tradeoff.key);
            return (
              <button
                key={tradeoff.key}
                type="button"
                onClick={() => toggle(tradeoff.key)}
                className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm transition-colors ${
                  active
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-card hover:border-primary/40"
                }`}
              >
                {active ? <Check className="h-4 w-4" /> : <Target className="h-4 w-4" />}
                {tradeoff.label}
              </button>
            );
          })}
        </div>
      </div>

      {availableAcceptances.length > 0 ? (
        <div className="rounded-xl border border-border bg-muted/25 p-4">
          <p className="mb-1 text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
            Tradeoffs you would accept
          </p>
          <p className="mb-3 text-sm text-muted-foreground">
            These only appear after you choose a goal. Use them when you would accept a cost to get that goal.
          </p>
          <div className="flex flex-wrap gap-2">
            {availableAcceptances.map((tradeoff) => {
              const active = accepted.includes(tradeoff.key);
              return (
                <button
                  key={tradeoff.key}
                  type="button"
                  title={tradeoff.description}
                  onClick={() => toggleAccepted(tradeoff.key)}
                  className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm transition-colors ${
                    active
                      ? "border-olive bg-olive text-primary-foreground"
                      : "border-border bg-card hover:border-olive/60"
                  }`}
                >
                  {active ? <Check className="h-4 w-4" /> : <Target className="h-4 w-4" />}
                  {tradeoff.label}
                </button>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
