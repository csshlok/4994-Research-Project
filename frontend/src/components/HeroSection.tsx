import { Button } from "@/components/ui/button";
import { ArrowDown } from "lucide-react";

interface HeroSectionProps {
  onAnalyzeClick: () => void;
}

export function HeroSection({ onAnalyzeClick }: HeroSectionProps) {
  return (
    <section className="relative min-h-[90vh] flex items-center justify-center section-padding overflow-hidden">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-20 left-10 w-72 h-72 bg-olive-muted rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-cream-dark rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: "1.5s" }} />
      </div>

      <div className="container-narrow relative z-10">
        <div className="text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/80 border border-border/50 mb-8 opacity-0 animate-fade-up">
            <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm font-medium text-muted-foreground">
              Research-grounded analysis
            </span>
          </div>

          {/* Main headline */}
          <h1 className="font-serif text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-semibold text-foreground leading-[1.1] mb-6 opacity-0 animate-fade-up stagger-1 text-balance">
            Understand companies through the lens of{" "}
            <span className="text-primary">human goals</span>
          </h1>

          {/* Subheadline */}
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed mb-10 opacity-0 animate-fade-up stagger-2 text-balance">
            We analyze employee reviews at scale, mapping experiences to seven fundamental social goals. 
            Discover how companies truly support or hinder what matters most to their people—beyond salary and perks.
          </p>

          {/* CTA */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 opacity-0 animate-fade-up stagger-3">
            <Button variant="hero" onClick={onAnalyzeClick}>
              Analyze a Company
            </Button>
            <Button variant="hero-outline" asChild>
              <a href="#domains">
                Learn the Framework
                <ArrowDown className="w-4 h-4" />
              </a>
            </Button>
          </div>

          {/* Trust indicator */}
          <p className="mt-12 text-sm text-muted-foreground opacity-0 animate-fade-up stagger-4">
            Grounded in behavioral science • No account required
          </p>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 opacity-0 animate-fade-up stagger-5">
        <div className="w-6 h-10 rounded-full border-2 border-border flex items-start justify-center p-2">
          <div className="w-1 h-2 bg-muted-foreground rounded-full animate-bounce" />
        </div>
      </div>
    </section>
  );
}
