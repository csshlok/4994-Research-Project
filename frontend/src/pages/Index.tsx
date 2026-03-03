import { useRef, useState } from "react";
import { HeroSection } from "@/components/HeroSection";
import { DomainsSection } from "@/components/DomainsSection";
import { CompanyInputSection } from "@/components/CompanyInputSection";
import { ProcessingAnimation } from "@/components/ProcessingAnimation";
import { ResultsPage } from "@/components/ResultsPage";
import {
  downloadJobFileText,
  getJobOutputs,
  getJobStatus,
  sleep,
  startPipelineJob,
} from "@/lib/backend-api";
import { AnalysisResult, buildAnalysisResult } from "@/lib/analysis";
import { toast } from "@/components/ui/use-toast";

type AppState = "landing" | "processing" | "results";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("landing");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [processingStatus, setProcessingStatus] = useState("Submitting analysis job...");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const inputRef = useRef<HTMLDivElement>(null);

  const scrollToInput = () => {
    inputRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleAnalyze = async (name: string) => {
    setAppState("processing");
    setAnalysis(null);
    setActiveJobId(null);
    setProcessingStatus("Submitting analysis job...");

    try {
      const run = await startPipelineJob(name);
      setActiveJobId(run.job_id);

      let finalStatus = null as Awaited<ReturnType<typeof getJobStatus>> | null;
      for (let attempt = 0; attempt < 360; attempt += 1) {
        const st = await getJobStatus(run.job_id);
        finalStatus = st;

        const stage = st.stage?.current ? `Stage: ${st.stage.current}` : "";
        const msg = st.message || stage || "Pipeline running";
        setProcessingStatus(msg);

        if (st.status === "succeeded") {
          break;
        }
        if (st.status === "failed") {
          throw new Error(st.message || "Pipeline failed.");
        }
        await sleep(2000);
      }

      if (!finalStatus || finalStatus.status !== "succeeded") {
        throw new Error("Pipeline timed out before completion.");
      }

      const outputs = await getJobOutputs(run.job_id);
      const reviewCsvText = await downloadJobFileText(run.job_id, "04_score/review_scores.csv");
      const companyCsvText = await downloadJobFileText(run.job_id, "04_score/company_scores.csv");

      const result = buildAnalysisResult({
        jobId: run.job_id,
        inputCompanyName: name,
        resolvedCompanyName: finalStatus.company_id_resolved,
        reviewCsvText,
        companyCsvText,
        outputFiles: outputs.files,
      });

      setAnalysis(result);
      setAppState("results");
    } catch (error) {
      const description = error instanceof Error ? error.message : "Unexpected error.";
      toast({
        title: "Analysis failed",
        description,
        variant: "destructive",
      });
      setActiveJobId(null);
      setAppState("landing");
    }
  };

  const handleBack = () => {
    setAppState("landing");
    setAnalysis(null);
    setActiveJobId(null);
    setProcessingStatus("Submitting analysis job...");
  };

  if (appState === "processing") {
    return <ProcessingAnimation statusMessage={processingStatus} jobId={activeJobId} />;
  }

  if (appState === "results" && analysis) {
    return <ResultsPage analysis={analysis} onBack={handleBack} />;
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
