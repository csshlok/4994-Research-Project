import { useState } from "react";
import { ArrowLeft, Calendar, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { MatchTradeoffControls } from "@/components/MatchTradeoffControls";
import {
  type MatchAcceptanceKey,
  type MatchInput,
  type MatchTradeoffKey,
  validateMatchInput,
} from "@/lib/matching";

interface MatchInputPageProps {
  title?: string;
  subtitle?: string;
  dateLabel: string;
  onBack: () => void;
  onSubmit: (input: MatchInput) => void;
}

export function MatchInputPage({
  title = "Tell us a bit about yourself",
  subtitle = "Describe the kind of workplace you want, then choose the tradeoffs that matter most.",
  dateLabel,
  onBack,
  onSubmit,
}: MatchInputPageProps) {
  const [narrative, setNarrative] = useState("");
  const [tradeoffs, setTradeoffs] = useState<MatchTradeoffKey[]>([]);
  const [acceptedTradeoffs, setAcceptedTradeoffs] = useState<MatchAcceptanceKey[]>([]);
  const [error, setError] = useState("");

  const handleSubmit = () => {
    const input = { narrative, tradeoffs, acceptedTradeoffs };
    const validationError = validateMatchInput(input);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError("");
    onSubmit(input);
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container-wide py-4 flex items-center justify-between">
          <Button variant="ghost" onClick={onBack} className="gap-2">
            <ArrowLeft className="w-4 h-4" />
            New Analysis
          </Button>
          <div className="text-sm text-muted-foreground flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            {dateLabel}
          </div>
        </div>
      </header>

      <main className="container-narrow py-12 md:py-16">
        <Card>
          <CardContent className="p-6 md:p-8">
            <p className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
              Behavioral Match
            </p>
            <h1 className="font-serif text-4xl font-semibold text-foreground md:text-5xl">
              {title}
            </h1>
            <p className="mt-4 max-w-2xl text-muted-foreground">{subtitle}</p>

            <div className="mt-8">
              <Textarea
                value={narrative}
                onChange={(event) => {
                  setNarrative(event.target.value);
                  setError("");
                }}
                className="min-h-[180px] resize-none bg-background text-base leading-relaxed"
                placeholder="Example: I want fair managers, strong learning, autonomy, and a calm culture that does not reward burnout or politics."
              />
            </div>

            <div className="mt-5">
              <MatchTradeoffControls
                selected={tradeoffs}
                accepted={acceptedTradeoffs}
                onChange={(next) => {
                  setTradeoffs(next);
                  setError("");
                }}
                onAcceptedChange={(next) => {
                  setAcceptedTradeoffs(next);
                  setError("");
                }}
              />
            </div>

            {error ? <p className="mt-3 text-sm text-red-500">{error}</p> : null}

            <div className="mt-7 flex justify-end">
              <Button onClick={handleSubmit} className="gap-2">
                <Sparkles className="h-4 w-4" />
                Match me
              </Button>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
