import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, ArrowRight } from "lucide-react";

interface CompanyInputSectionProps {
  onSubmit: (companyName: string) => void;
}

export function CompanyInputSection({ onSubmit }: CompanyInputSectionProps) {
  const [companyName, setCompanyName] = useState("");
  const [isFocused, setIsFocused] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (companyName.trim()) {
      onSubmit(companyName.trim());
    }
  };

  return (
    <section id="analyze" className="section-padding">
      <div className="container-narrow">
        <div className="bg-card rounded-3xl border border-border/50 shadow-elevated p-8 md:p-12">
          <div className="text-center max-w-xl mx-auto mb-10">
            <span className="inline-block text-sm font-medium text-primary uppercase tracking-wider mb-4">
              Start Analysis
            </span>
            <h2 className="font-serif text-3xl md:text-4xl font-semibold text-foreground mb-4">
              Analyze a Company
            </h2>
            <p className="text-muted-foreground leading-relaxed">
              Enter a company name to receive a comprehensive analysis of employee experiences
              mapped to the seven social domains.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="max-w-lg mx-auto">
            <div
              className={`relative flex items-center gap-3 p-2 rounded-2xl border-2 transition-all duration-300 bg-background ${
                isFocused
                  ? "border-primary shadow-soft"
                  : "border-border hover:border-border/80"
              }`}
            >
              <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-secondary text-primary">
                <Search className="w-5 h-5" />
              </div>
              <Input
                type="text"
                value={companyName}
                onChange={(e) => setCompanyName(e.target.value)}
                onFocus={() => setIsFocused(true)}
                onBlur={() => setIsFocused(false)}
                placeholder="Enter company name..."
                className="flex-1 border-0 bg-transparent text-lg placeholder:text-muted-foreground/60 focus-visible:ring-0 focus-visible:ring-offset-0 h-12"
                maxLength={100}
              />
              <Button
                type="submit"
                variant="hero"
                disabled={!companyName.trim()}
                className="h-12 px-6"
              >
                <span className="hidden sm:inline">Run Analysis</span>
                <ArrowRight className="w-5 h-5" />
              </Button>
            </div>

            <p className="text-center text-sm text-muted-foreground mt-4">
              Analysis typically takes 2-3 minutes depending on review volume
            </p>
          </form>
        </div>
      </div>
    </section>
  );
}
