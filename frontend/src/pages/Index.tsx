import { useEffect, useRef, useState } from "react";
import { HeroSection } from "@/components/HeroSection";
import { SummarySection } from "@/components/SummarySection";
import { DomainsSection } from "@/components/DomainsSection";
import { CompanyInputSection } from "@/components/CompanyInputSection";
import { ProcessingAnimation } from "@/components/ProcessingAnimation";
import { ResultsPage } from "@/components/ResultsPage";
import { ComparisonResultsPage } from "@/components/ComparisonResultsPage";
import { MatchInputPage } from "@/components/MatchInputPage";
import { MatchResultsView } from "@/components/MatchResultsView";
import { Button } from "@/components/ui/button";
import {
  downloadScoredCompanyFileText,
  generateComparisonRagSummary,
  generateMatchTopSummary,
  getScoredCompanyDownloadUrl,
  getScoredCompanyOutputs,
  getScoredCompanyRag,
  interpretMatchProfile,
  sleep,
  type ComparisonRagSummary,
  type MatchTopSummary,
} from "@/lib/backend-api";
import { AnalysisResult, buildAnalysisResult } from "@/lib/analysis";
import {
  buildCompanyComparisonMetric,
  type CompanyComparisonMetric,
} from "@/lib/comparison";
import {
  buildMatchProfile,
  mergeBehaviorProfile,
  scoreCompanyMatch,
  type CompanyMatchResult,
  type MatchInput,
  type MatchProfile,
} from "@/lib/matching";
import { toast } from "@/components/ui/use-toast";
import { COMPANY_OPTIONS } from "@/lib/company-options";
import { ArrowLeft, Calendar } from "lucide-react";

type AppState =
  | "landing"
  | "processing"
  | "results"
  | "comparison"
  | "matches"
  | "singleMatchInput"
  | "comparisonMatchInput";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("landing");
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [comparison, setComparison] = useState<CompanyComparisonMetric[]>([]);
  const [comparisonRag, setComparisonRag] = useState<ComparisonRagSummary | null>(null);
  const [matchProfile, setMatchProfile] = useState<MatchProfile | null>(null);
  const [matchResults, setMatchResults] = useState<CompanyMatchResult[]>([]);
  const [matchTitle, setMatchTitle] = useState("Your matches");
  const [matchTopSummary, setMatchTopSummary] = useState<MatchTopSummary | null>(null);
  const [processingStatus, setProcessingStatus] = useState("Preparing your analysis...");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [compareSeed, setCompareSeed] = useState<string[]>([]);
  const [comparisonMatchTargetId, setComparisonMatchTargetId] = useState<string | null>(null);
  const [inputResetKey, setInputResetKey] = useState(0);
  const inputRef = useRef<HTMLDivElement>(null);

  const scrollToInput = () => {
    inputRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const handleAnalyze = async (name: string) => {
    setAppState("processing");
    setAnalysis(null);
    setMatchProfile(null);
    setMatchResults([]);
    setMatchTopSummary(null);
    setActiveJobId(null);
    setProcessingStatus("Preparing your company analysis...");

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
    } catch (error) {
      const description =
        error instanceof Error && error.message.includes("404")
          ? `No cached analysis is available for "${name}" yet.`
          : error instanceof Error
            ? error.message
            : "Unexpected error.";
      toast({
        title: "Analysis failed",
        description,
        variant: "destructive",
      });
      setActiveJobId(null);
      setAppState("landing");
    }
  };

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const company = params.get("company");
    if (!company) return;
    window.history.replaceState({}, "", window.location.pathname);
    void handleAnalyze(company);
  }, []);

  const handleCompare = async (companyNames: string[]) => {
    const uniqueCompanies = Array.from(new Set(companyNames)).slice(0, 3);
    setAppState("processing");
    setAnalysis(null);
    setComparison([]);
    setComparisonRag(null);
    setMatchProfile(null);
    setMatchResults([]);
    setMatchTopSummary(null);
    setActiveJobId(null);
    setProcessingStatus("Preparing your company comparison...");

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

  const buildCompanyMetric = async (companyName: string) => {
    const [companyCsvText, topicCsvText] = await Promise.all([
      downloadScoredCompanyFileText(companyName, "company_scores.csv"),
      downloadScoredCompanyFileText(companyName, "topic_summary.csv"),
    ]);
    return buildCompanyComparisonMetric(companyName, companyCsvText, topicCsvText);
  };

  const buildInterpretedMatchProfile = async (input: MatchInput) => {
    const localProfile = buildMatchProfile(input);
    try {
      const interpreted = await interpretMatchProfile({
        narrative: input.narrative,
        tradeoffs: input.tradeoffs,
        accepted_tradeoffs: input.acceptedTradeoffs,
        local_profile: localProfile,
      });
      return mergeBehaviorProfile(localProfile, {
        summary: interpreted.summary,
        likelyMotivators: interpreted.likely_motivators,
        coreNeeds: interpreted.core_needs,
        riskSensitivities: interpreted.risk_sensitivities,
        workStyle: interpreted.work_style,
        confidence: interpreted.confidence,
        source: interpreted.source === "gemini" ? "llm" : "fallback",
      });
    } catch {
      return localProfile;
    }
  };

  const buildTopMatchSummary = async (
    profile: MatchProfile,
    results: CompanyMatchResult[]
  ): Promise<MatchTopSummary | null> => {
    const top = results[0];
    if (!top) return null;

    try {
      return await generateMatchTopSummary({
        profile: profile.behaviorProfile,
        top_match: {
          company: top.company.label,
          score: top.score,
          domain_alignment: top.domainAlignment,
          theme_bonus: top.themeBonus,
          risk_penalty: top.riskPenalty,
          confidence: top.confidence,
          why_fit: top.whyFit,
          watch_outs: top.watchOuts,
          domain_scores: top.company.domains,
        },
        evidence: top.evidence.slice(0, 5).map((item) => ({
          mode: item.mode,
          label: item.label,
          text: item.text,
          role: item.role,
          date: item.date,
        })),
      });
    } catch {
      return null;
    }
  };

  const handleMatch = async (input: MatchInput) => {
    setAppState("processing");
    setAnalysis(null);
    setComparison([]);
    setComparisonRag(null);
    setMatchResults([]);
    setMatchTopSummary(null);
    setActiveJobId(null);
    setProcessingStatus("Building your match profile...");

    try {
      const profile = await buildInterpretedMatchProfile(input);
      setMatchProfile(profile);
      setMatchTitle("Your matches");
      const metricSettled = await Promise.allSettled(
        COMPANY_OPTIONS.map(async (company) => buildCompanyMetric(company.value))
      );
      const metrics = metricSettled
        .filter((result): result is PromiseFulfilledResult<CompanyComparisonMetric> => result.status === "fulfilled")
        .map((result) => result.value);

      setProcessingStatus("Ranking companies against your profile...");
      const preliminary = metrics
        .map((metric) => scoreCompanyMatch(profile, metric))
        .sort((a, b) => b.score - a.score)
        .slice(0, 4);

      setProcessingStatus("Collecting review evidence for your matches...");
      const enriched = await Promise.all(
        preliminary.map(async (result) => {
          try {
            const rag = await getScoredCompanyRag(result.company.id);
            return scoreCompanyMatch(profile, result.company, rag);
          } catch {
            return result;
          }
        })
      );

      const sorted = enriched.sort((a, b) => b.score - a.score);
      setMatchResults(sorted);
      const summary = await buildTopMatchSummary(profile, sorted);
      setMatchTopSummary(summary);
      setAppState("matches");
    } catch (error) {
      const description = error instanceof Error ? error.message : "Unexpected error.";
      toast({
        title: "Match failed",
        description,
        variant: "destructive",
      });
      setAppState("landing");
    }
  };

  const buildMetricFromAnalysis = (source: AnalysisResult): CompanyComparisonMetric => ({
    id: source.companyId,
    label: source.companyName,
    overall: source.overallScore,
    reviews: source.reviewCount,
    domains: Object.fromEntries(
      source.domainScores.map((domain) => [domain.code, Math.round(domain.scorePct)])
    ),
    topics: source.topicClusters.map((cluster) => ({
      mode: cluster.mode,
      label: cluster.label,
      domain: cluster.domain,
      reviewCount: cluster.reviewCount,
      signal: cluster.signal,
      x: cluster.x,
      y: cluster.y,
      terms: cluster.terms,
    })),
  });

  const handleSingleCompanyMatch = async (input: MatchInput) => {
    if (!analysis) return;
    const currentAnalysis = analysis;
    setAppState("processing");
    setProcessingStatus("Building your match profile...");
    setMatchResults([]);
    setMatchProfile(null);
    setMatchTopSummary(null);

    try {
      const profile = await buildInterpretedMatchProfile(input);
      const rag = await getScoredCompanyRag(currentAnalysis.companyId).catch(() => currentAnalysis.rag);
      const result = scoreCompanyMatch(profile, buildMetricFromAnalysis(currentAnalysis), rag);
      setMatchProfile(profile);
      setMatchTitle(`Your match to ${currentAnalysis.companyName}`);
      setMatchResults([result]);
      const summary = await buildTopMatchSummary(profile, [result]);
      setMatchTopSummary(summary);
      setAppState("matches");
    } catch (error) {
      const description = error instanceof Error ? error.message : "Unexpected error.";
      toast({
        title: "Match failed",
        description,
        variant: "destructive",
      });
      setAppState("results");
    }
  };

  const handleComparisonMatch = async (input: MatchInput) => {
    const targetMetrics = comparisonMatchTargetId
      ? comparison.filter((metric) => metric.id === comparisonMatchTargetId)
      : comparison;

    if (targetMetrics.length === 0) {
      setAppState("comparison");
      return;
    }

    setAppState("processing");
    setProcessingStatus(
      comparisonMatchTargetId
        ? "Building your company-specific fit..."
        : "Ranking your fit across the compared companies..."
    );
    setMatchResults([]);
    setMatchProfile(null);
    setMatchTopSummary(null);

    try {
      const profile = await buildInterpretedMatchProfile(input);
      const enriched = await Promise.all(
        targetMetrics.map(async (metric) => {
          try {
            const rag = await getScoredCompanyRag(metric.id);
            return scoreCompanyMatch(profile, metric, rag);
          } catch {
            return scoreCompanyMatch(profile, metric);
          }
        })
      );
      const sorted = enriched.sort((a, b) => b.score - a.score);
      setMatchProfile(profile);
      setMatchTitle(
        comparisonMatchTargetId
          ? `Your match to ${targetMetrics[0].label}`
          : "Your fit across these companies"
      );
      setMatchResults(sorted);
      setMatchTopSummary(await buildTopMatchSummary(profile, sorted));
      setAppState("comparison");
    } catch (error) {
      const description = error instanceof Error ? error.message : "Unexpected error.";
      toast({
        title: "Match failed",
        description,
        variant: "destructive",
      });
      setAppState("comparison");
    }
  };

  const openCompanyAnalysisInNewTab = (companyId: string) => {
    const url = `${window.location.origin}${window.location.pathname}?company=${encodeURIComponent(companyId)}`;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  const handleBack = () => {
    setAppState("landing");
    setAnalysis(null);
    setComparison([]);
    setComparisonRag(null);
    setMatchProfile(null);
    setMatchResults([]);
    setMatchTopSummary(null);
    setActiveJobId(null);
    setCompareSeed([]);
    setComparisonMatchTargetId(null);
    setInputResetKey((key) => key + 1);
    setProcessingStatus("Preparing your analysis...");
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
        onMatchStart={() => setAppState("singleMatchInput")}
      />
    );
  }

  if (appState === "comparison" && comparison.length > 0) {
    return (
      <ComparisonResultsPage
        metrics={comparison}
        ragSummary={comparisonRag}
        matchProfile={matchProfile}
        matchResults={matchResults}
        matchTitle={matchTitle}
        matchTopSummary={matchTopSummary}
        onBack={handleBack}
        onStartMatch={(companyId) => {
          setComparisonMatchTargetId(companyId || null);
          setAppState("comparisonMatchInput");
        }}
        onOpenCompany={openCompanyAnalysisInNewTab}
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

  if (appState === "singleMatchInput" && analysis) {
    return (
      <MatchInputPage
        dateLabel={analysis.analysisDate}
        onBack={() => setAppState("results")}
        onSubmit={handleSingleCompanyMatch}
      />
    );
  }

  if (appState === "comparisonMatchInput") {
    const selectedMetric = comparisonMatchTargetId
      ? comparison.find((metric) => metric.id === comparisonMatchTargetId)
      : null;
    return (
      <MatchInputPage
        title="Tell us a bit about yourself"
        subtitle={
          selectedMetric
            ? `Describe what matters to you at work, then we will match you to ${selectedMetric.label}.`
            : "Describe what matters to you at work, then we will rank your fit across the compared companies."
        }
        dateLabel={new Date().toLocaleDateString("en-US", {
          year: "numeric",
          month: "long",
          day: "numeric",
        })}
        onBack={() => setAppState("comparison")}
        onSubmit={handleComparisonMatch}
      />
    );
  }

  if (appState === "matches" && matchProfile && matchResults.length > 0) {
    const todayLabel = new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });

    return (
      <div className="min-h-screen bg-background">
        <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
          <div className="container-wide py-4 flex items-center justify-between">
            <Button variant="ghost" onClick={handleBack} className="gap-2">
              <ArrowLeft className="h-4 w-4" />
              New Analysis
            </Button>
            <div className="text-sm text-muted-foreground flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              {todayLabel}
            </div>
          </div>
        </header>
        <main className="container-wide py-10 md:py-14">
          <MatchResultsView
            title={matchTitle}
            profile={matchProfile}
            results={matchResults}
            onOpenCompany={openCompanyAnalysisInNewTab}
            topSummary={matchTopSummary}
          />
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <HeroSection onAnalyzeClick={scrollToInput} />
      <SummarySection />
      <DomainsSection />
      <div ref={inputRef}>
        <CompanyInputSection
          key={`${inputResetKey}-${compareSeed.join("|") || "empty"}`}
          onSubmit={handleAnalyze}
          onCompareSubmit={handleCompare}
          onMatchSubmit={handleMatch}
          initialCompareCompanies={compareSeed}
        />
      </div>
      
      <footer className="py-12 border-t border-border">
        <div className="container-narrow text-center">
          <p className="text-sm text-muted-foreground">
            Built on behavioral science research - Designed for researchers, students, and curious minds
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
