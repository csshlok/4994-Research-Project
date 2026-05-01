import { useRef, useState } from "react";
import { HeroSection } from "@/components/HeroSection";
import { SummarySection } from "@/components/SummarySection";
import { DomainsSection } from "@/components/DomainsSection";
import { CompanyInputSection } from "@/components/CompanyInputSection";
import { ProcessingAnimation } from "@/components/ProcessingAnimation";
import { ResultsPage } from "@/components/ResultsPage";
import { ComparisonResultsPage } from "@/components/ComparisonResultsPage";
import {
  downloadScoredCompanyFileText,
  generateComparisonRagSummary,
  downloadJobFileText,
  getScoredCompanyDownloadUrl,
  getScoredCompanyOutputs,
  getScoredCompanyRag,
  getJobOutputs,
  getJobStatus,
  sleep,
  startPipelineJob,
  type ComparisonRagSummary,
} from "@/lib/backend-api";
import { AnalysisResult, buildAnalysisResult } from "@/lib/analysis";
import {
  buildCompanyComparisonMetric,
  type CompanyComparisonMetric,
} from "@/lib/comparison";
import { toast } from "@/components/ui/use-toast";

type AppState = "landing" | "processing" | "results" | "comparison";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("landing");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [comparison, setComparison] = useState<CompanyComparisonMetric[]>([]);
  const [comparisonRag, setComparisonRag] = useState<ComparisonRagSummary | null>(null);
  const [processingStatus, setProcessingStatus] = useState("Submitting analysis job...");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [compareSeed, setCompareSeed] = useState<string[]>([]);
  const inputRef = useRef<HTMLDivElement>(null);

  const scrollToInput = () => {
    inputRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleAnalyze = async (name: string) => {
    setAppState("processing");
    setAnalysis(null);
    setActiveJobId(null);
    setProcessingStatus("Loading precomputed company scores...");

    try {
      try {
        const cachedOutputs = await getScoredCompanyOutputs(name);
        const reviewCsvText = await downloadScoredCompanyFileText(
          cachedOutputs.company_id,
          "review_scores.csv"
        );
        const companyCsvText = await downloadScoredCompanyFileText(
          cachedOutputs.company_id,
          "company_scores.csv"
        );
        const topicCsvText = cachedOutputs.files.includes("topic_summary.csv")
          ? await downloadScoredCompanyFileText(cachedOutputs.company_id, "topic_summary.csv")
          : undefined;
        const rag = cachedOutputs.files.includes("rag_summary.json")
          ? await getScoredCompanyRag(cachedOutputs.company_id)
          : undefined;

        const result = buildAnalysisResult({
          jobId: `cached-${cachedOutputs.company_id}`,
          inputCompanyName: name,
          resolvedCompanyName: cachedOutputs.company_id,
          reviewCsvText,
          companyCsvText,
          topicCsvText,
          ragSummary: rag?.summary,
          ragClusters: rag?.clusters,
          ragInsights: rag?.insights,
          outputFiles: cachedOutputs.files,
          downloads: {
            cleanedReviews: cachedOutputs.files.includes("cleaned_reviews.csv")
              ? getScoredCompanyDownloadUrl(cachedOutputs.company_id, "cleaned_reviews.csv")
              : undefined,
            reviewScores: getScoredCompanyDownloadUrl(cachedOutputs.company_id, "review_scores.csv"),
            companyScores: getScoredCompanyDownloadUrl(cachedOutputs.company_id, "company_scores.csv"),
            topicSummary: cachedOutputs.files.includes("topic_summary.csv")
              ? getScoredCompanyDownloadUrl(cachedOutputs.company_id, "topic_summary.csv")
              : undefined,
            topicAssignments: cachedOutputs.files.includes("topic_assignments.csv")
              ? getScoredCompanyDownloadUrl(cachedOutputs.company_id, "topic_assignments.csv")
              : undefined,
          },
        });

        await sleep(10000);
        setAnalysis(result);
        setAppState("results");
        return;
      } catch {
        setProcessingStatus("No precomputed scores found. Submitting analysis job...");
      }

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

      await sleep(10000);
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

  const handleCompare = async (companyNames: string[]) => {
    const uniqueCompanies = Array.from(new Set(companyNames)).slice(0, 3);
    setAppState("processing");
    setAnalysis(null);
    setComparison([]);
    setComparisonRag(null);
    setActiveJobId(null);
    setProcessingStatus("Preparing cached company comparison...");

    try {
      const metrics = await Promise.all(
        uniqueCompanies.map(async (companyName) => {
          const [companyCsvText, topicCsvText] = await Promise.all([
            downloadScoredCompanyFileText(companyName, "company_scores.csv"),
            downloadScoredCompanyFileText(companyName, "topic_summary.csv"),
          ]);
          return buildCompanyComparisonMetric(companyName, companyCsvText, topicCsvText);
        })
      );
      setProcessingStatus("Generating comparison summary...");
      const ragSummary = await generateComparisonRagSummary(uniqueCompanies);
      await sleep(10000);
      setComparison(metrics);
      setComparisonRag(ragSummary);
      setAppState("comparison");
    } catch (error) {
      const description = error instanceof Error ? error.message : "Unexpected error.";
      toast({
        title: "Comparison failed",
        description,
        variant: "destructive",
      });
      setAppState("landing");
    }
  };

  const handleBack = () => {
    setAppState("landing");
    setAnalysis(null);
    setComparison([]);
    setComparisonRag(null);
    setActiveJobId(null);
    setCompareSeed([]);
    setProcessingStatus("Submitting analysis job...");
  };

  const handleCompareFromResults = (companyId: string) => {
    setCompareSeed([companyId]);
    setAnalysis(null);
    setAppState("landing");
    window.setTimeout(() => {
      inputRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 50);
  };

  if (appState === "processing") {
    return <ProcessingAnimation statusMessage={processingStatus} jobId={activeJobId} />;
  }

  if (appState === "results" && analysis) {
    return (
      <ResultsPage
        analysis={analysis}
        onBack={handleBack}
        onCompare={handleCompareFromResults}
      />
    );
  }

  if (appState === "comparison" && comparison.length > 0) {
    return (
      <ComparisonResultsPage
        metrics={comparison}
        ragSummary={comparisonRag}
        onBack={handleBack}
        onAddComparison={(seedCompanyId) => {
          setCompareSeed(seedCompanyId ? [seedCompanyId] : []);
          setComparison([]);
          setComparisonRag(null);
          setAppState("landing");
          window.setTimeout(() => {
            inputRef.current?.scrollIntoView({ behavior: "smooth" });
          }, 50);
        }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <HeroSection onAnalyzeClick={scrollToInput} />
      <SummarySection />
      <DomainsSection />
      <div ref={inputRef}>
        <CompanyInputSection
          key={compareSeed.join("|") || "empty"}
          onSubmit={handleAnalyze}
          onCompareSubmit={handleCompare}
          initialCompareCompanies={compareSeed}
        />
      </div>
      
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
