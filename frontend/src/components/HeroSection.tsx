import { Button } from "@/components/ui/button";
import { ArrowDown } from "lucide-react";

interface HeroSectionProps {
  onAnalyzeClick: () => void;
}

export function HeroSection({ onAnalyzeClick }: HeroSectionProps) {
  return (
    <section className="relative min-h-[90vh] flex items-center justify-center section-padding overflow-hidden">
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-20 left-10 h-72 w-72 rounded-full bg-olive-muted blur-3xl animate-pulse-slow" />
        <div
          className="absolute bottom-20 right-10 h-96 w-96 rounded-full bg-cream-dark blur-3xl animate-pulse-slow"
          style={{ animationDelay: "1.5s" }}
        />
      </div>

      <div className="container-narrow relative z-10">
        <div className="text-center">
          <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-border/50 bg-secondary/80 px-4 py-2 opacity-0 animate-fade-up">
            <div className="h-2 w-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm font-medium text-muted-foreground">
              Research-grounded analysis
            </span>
          </div>

          <h1 className="mb-6 font-serif text-4xl font-semibold leading-[1.1] text-foreground opacity-0 animate-fade-up stagger-1 text-balance sm:text-5xl md:text-6xl lg:text-7xl">
            Understand companies through the lens of{" "}
            <span className="text-primary">human goals</span>
          </h1>

          <p className="mx-auto mb-10 max-w-2xl text-balance text-lg leading-relaxed text-muted-foreground opacity-0 animate-fade-up stagger-2 md:text-xl">
            We analyze employee reviews at scale, mapping experiences to five
            fundamental workplace goals. Discover how companies truly support or
            hinder what matters most to their people beyond salary and perks.
          </p>

          <div className="flex flex-col items-center justify-center gap-4 opacity-0 animate-fade-up stagger-3 sm:flex-row">
            <Button variant="hero" onClick={onAnalyzeClick}>
              Analyze a Company
            </Button>
            <Button variant="hero-outline" asChild>
              <a href="#domains">
                Learn the Framework
                <ArrowDown className="h-4 w-4" />
              </a>
            </Button>
          </div>

          <p className="mt-12 text-sm text-muted-foreground opacity-0 animate-fade-up stagger-4">
            Grounded in behavioral science • No account required
          </p>
        </div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2 opacity-0 animate-fade-up stagger-5">
        <div className="flex h-10 w-6 items-start justify-center rounded-full border-2 border-border p-2">
          <div className="h-2 w-1 rounded-full bg-muted-foreground animate-bounce" />
        </div>
      </div>
    </section>
  );
}
