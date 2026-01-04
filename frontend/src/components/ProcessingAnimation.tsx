import { useEffect, useState } from "react";

const processingSteps = [
  "Reading employee reviews...",
  "Extracting sentiment patterns...",
  "Mapping to social domains...",
  "Analyzing goal fulfillment...",
  "Calculating aggregated scores...",
  "Generating visualizations...",
  "Preparing your report...",
];

export function ProcessingAnimation() {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % processingSteps.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="text-center max-w-lg">
        {/* Animated circles */}
        <div className="relative w-48 h-48 mx-auto mb-12">
          {/* Outer rotating ring */}
          <div className="absolute inset-0 rounded-full border-2 border-dashed border-olive-muted animate-spin" style={{ animationDuration: "20s" }} />
          
          {/* Middle pulsing ring */}
          <div className="absolute inset-4 rounded-full border border-primary/30 animate-pulse" />
          
          {/* Inner glow */}
          <div className="absolute inset-8 rounded-full bg-primary/10 animate-pulse" style={{ animationDelay: "0.5s" }} />
          
          {/* Center orb */}
          <div className="absolute inset-16 rounded-full bg-gradient-to-br from-primary to-forest-light shadow-elevated flex items-center justify-center">
            <div className="w-8 h-8 rounded-full bg-primary-foreground/20 animate-ping" />
          </div>

          {/* Floating particles */}
          {[...Array(6)].map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 rounded-full bg-primary/60 animate-float"
              style={{
                top: `${20 + Math.sin(i * 60 * Math.PI / 180) * 40}%`,
                left: `${50 + Math.cos(i * 60 * Math.PI / 180) * 40}%`,
                animationDelay: `${i * 0.3}s`,
              }}
            />
          ))}
        </div>

        {/* Status text */}
        <div className="space-y-4">
          <p className="text-xl font-serif font-medium text-foreground animate-fade-in" key={currentStep}>
            {processingSteps[currentStep]}
          </p>
          <p className="text-muted-foreground">
            This usually takes a few moments. We're analyzing employee experiences across multiple dimensions.
          </p>
        </div>

        {/* Progress dots */}
        <div className="flex items-center justify-center gap-2 mt-8">
          {processingSteps.map((_, i) => (
            <div
              key={i}
              className={`w-2 h-2 rounded-full transition-all duration-300 ${
                i === currentStep
                  ? "bg-primary w-6"
                  : i < currentStep
                  ? "bg-primary/60"
                  : "bg-border"
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
