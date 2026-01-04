import { useState, useRef } from "react";
import { HeroSection } from "@/components/HeroSection";
import { DomainsSection } from "@/components/DomainsSection";
import { CompanyInputSection } from "@/components/CompanyInputSection";
import { ProcessingAnimation } from "@/components/ProcessingAnimation";
import { ResultsPage } from "@/components/ResultsPage";

type AppState = "landing" | "processing" | "results";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("landing");
  const [companyName, setCompanyName] = useState("");
  const inputRef = useRef<HTMLDivElement>(null);

  const scrollToInput = () => {
    inputRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleAnalyze = (name: string) => {
    setCompanyName(name);
    setAppState("processing");
    
    // Simulate processing time (3-5 seconds for demo)
    setTimeout(() => {
      setAppState("results");
    }, 4000);
  };

  const handleBack = () => {
    setAppState("landing");
    setCompanyName("");
  };

  if (appState === "processing") {
    return <ProcessingAnimation />;
  }

  if (appState === "results") {
    return <ResultsPage companyName={companyName} onBack={handleBack} />;
  }

  return (
    <div className="min-h-screen bg-background">
      <HeroSection onAnalyzeClick={scrollToInput} />
      <DomainsSection />
      <div ref={inputRef}>
        <CompanyInputSection onSubmit={handleAnalyze} />
      </div>
      
      {/* Footer */}
      <footer className="py-12 border-t border-border">
        <div className="container-narrow text-center">
          <p className="text-sm text-muted-foreground">
            Built on behavioral science research • Designed for researchers, students, and curious minds
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
