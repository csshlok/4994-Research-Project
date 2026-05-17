import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { getScoredCompanyRag } from "@/lib/backend-api";
import { AnalysisResult, parseRagEvidence, type AnalysisEvidence } from "@/lib/analysis";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Download,
  FileSpreadsheet,
  BarChart3,
  ArrowLeft,
  Calendar,
  FileText,
  TrendingUp,
  TrendingDown,
  Sparkles,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from "recharts";

interface ResultsPageProps {
  analysis: AnalysisResult;
  onBack: () => void;
  onCompare?: (companyId: string) => void;
  onMatchStart?: () => void;
}

function DownloadCard({
  title,
  subtitle,
  href,
  icon,
}: {
  title: string;
  subtitle: string;
  href?: string;
  icon: ReactNode;
}) {
  const disabled = !href;

  return (
    <Card className="group hover:shadow-elevated transition-shadow">
      <CardContent className="pt-6 text-center">
        <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center mx-auto mb-4 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
          {icon}
        </div>
        <h3 className="font-medium mb-1">{title}</h3>
        <p className="text-sm text-muted-foreground mb-4">{subtitle}</p>
        {disabled ? (
          <Button variant="outline" size="sm" disabled className="gap-2">
            <Download className="w-4 h-4" />
            Unavailable
          </Button>
        ) : (
          <Button variant="outline" size="sm" asChild className="gap-2">
            <a href={href} target="_blank" rel="noreferrer">
              <Download className="w-4 h-4" />
              CSV
            </a>
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

const CLUSTER_COLORS = [
  "hsl(var(--primary))",
  "hsl(var(--olive))",
  "hsl(var(--destructive))",
  "hsl(var(--forest-light))",
  "hsl(var(--warm-gray))",
];

const POSITIVE_EVIDENCE_CUES = [
  "supportive",
  "safe",
  "valued",
  "respect",
  "growth",
  "opportunity",
  "opportunities",
  "inclusive",
  "coaching",
  "training",
  "clear expectations",
  "rewarding",
  "team",
  "benefits",
  "good pay",
  "flexible",
  "balance",
  "trust",
  "appreciation",
  "recognition",
];

const NEGATIVE_EVIDENCE_CUES = [
  "bad",
  "can't",
  "cannot",
  "no ",
  "not ",
  "guilty",
  "understaffed",
  "worn out",
  "burnout",
  "suffer",
  "challenging",
  "toxic",
  "favoritism",
  "sexism",
  "racism",
  "exploited",
  "unappreciated",
  "unsafe",
  "less than",
  "long hours",
  "work life balance",
  "poor management",
  "micromanagement",
];

function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

function countCueHits(sentence: string, cues: string[]): number {
  const normalized = sentence.toLowerCase();
  return cues.filter((cue) => normalized.includes(cue)).length;
}

function scoreEvidenceSentence(sentence: string, item: AnalysisEvidence): number {
  const normalized = sentence.toLowerCase();
  const termScore = item.terms.reduce((score, term) => {
    const clean = term.toLowerCase().trim();
    if (!clean || !normalized.includes(clean)) return score;
    return score + Math.max(1, Math.min(4, Math.ceil(clean.length / 12)));
  }, 0);
  const labelScore = normalized.includes(item.label.toLowerCase()) ? 2 : 0;
  const domainScore = item.domain && normalized.includes(item.domain.toLowerCase()) ? 1 : 0;
  const positiveScore = countCueHits(sentence, POSITIVE_EVIDENCE_CUES);
  const negativeScore = countCueHits(sentence, NEGATIVE_EVIDENCE_CUES);

  if (item.mode === "fulfillment") {
    return termScore + labelScore + domainScore + positiveScore * 2 - negativeScore * 3;
  }

  return termScore + labelScore + domainScore + negativeScore * 2 - positiveScore;
}

function selectCompanyEvidenceSentences(item: AnalysisEvidence): string[] {
  const sentences = splitSentences(item.text);
  const scored = sentences
    .map((sentence, index) => ({ sentence, index, score: scoreEvidenceSentence(sentence, item) }))
    .filter((row) => row.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, 2)
    .sort((a, b) => a.index - b.index)
    .map((row) => row.sentence);

  if (scored.length) {
    return scored;
  }

  const preferredFallback = sentences.filter((sentence) =>
    item.mode === "fulfillment"
      ? countCueHits(sentence, NEGATIVE_EVIDENCE_CUES) === 0
      : countCueHits(sentence, NEGATIVE_EVIDENCE_CUES) > 0
  );

  return (preferredFallback.length ? preferredFallback : sentences).slice(0, 2);
}

function CompanyEvidenceCard({ item }: { item: AnalysisEvidence }) {
  const selectedSentences = selectCompanyEvidenceSentences(item);
  const selectedSet = new Set(selectedSentences);
  const fullSentences = splitSentences(item.text);
  const explanation =
    item.mode === "fulfillment"
      ? `This is company-level strength evidence for ${item.label.toLowerCase()}: the excerpt describes conditions employees say are working well.`
      : `This is company-level risk evidence for ${item.label.toLowerCase()}: the excerpt describes a concrete friction employees report.`;

  return (
    <div className="rounded-lg border border-border p-5">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <Badge variant={item.mode === "fulfillment" ? "secondary" : "outline"}>
          {item.mode === "fulfillment" ? "fulfillment" : "hindrance"}
        </Badge>
        <p className="text-sm font-medium">{item.label}</p>
        {item.domain ? <span className="text-xs text-muted-foreground">{item.domain}</span> : null}
      </div>

      <div className="rounded-xl border border-primary/20 bg-primary/5 p-4">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-primary">
          Highlighted company context
        </p>
        <div className="space-y-2">
          {selectedSentences.map((sentence) => (
            <p key={sentence} className="text-sm leading-relaxed text-foreground">
              {sentence}
            </p>
          ))}
        </div>
        <p className="mt-3 text-xs leading-5 text-muted-foreground">{explanation}</p>
      </div>

      <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
        {fullSentences.map((sentence, index) => {
          const isSelected = selectedSet.has(sentence);
          return (
            <span
              key={`${sentence}-${index}`}
              className={isSelected ? "rounded bg-primary/10 px-1 text-foreground" : undefined}
            >
              {sentence}
              {index < fullSentences.length - 1 ? " " : ""}
            </span>
          );
        })}
      </p>

      {item.role || item.date || item.rating !== undefined ? (
        <p className="mt-3 text-xs text-muted-foreground">
          {[item.role, item.date?.slice(0, 10), item.rating !== undefined ? `${item.rating}/5` : undefined]
            .filter(Boolean)
            .join(" | ")}
        </p>
      ) : null}
    </div>
  );
}

export function ResultsPage({ analysis, onBack, onCompare, onMatchStart }: ResultsPageProps) {
  const [clusterMode, setClusterMode] = useState<"fulfillment" | "hindrance">("fulfillment");
  const [remoteReviewEvidence, setRemoteReviewEvidence] = useState<AnalysisEvidence[]>([]);
  const domainScores = analysis.domainScores;
  const overallScore = analysis.overallScore;
  const strongestDomain = analysis.strongestDomain;
  const weakestDomain = analysis.weakestDomain;
  const reviewCount = analysis.reviewCount;
  const analysisDate = analysis.analysisDate;
  const activeClusters = useMemo(
    () => analysis.topicClusters.filter((cluster) => cluster.mode === clusterMode),
    [analysis.topicClusters, clusterMode]
  );

  const barData = domainScores.map((d) => ({
    domain: d.domain,
    fulfillment: d.fulfillment,
    hindrance: d.hindrance,
    hindranceNeg: -d.hindrance,
  }));

  const radarData = domainScores.map((d) => ({
    domain: d.domain,
    score: d.scorePct,
  }));

const maxEvidence = Math.max(
  1,
  ...domainScores.map((d) => Math.max(d.fulfillment, d.hindrance))
);

const getRadarScaleMax = (value: number) => {
  const whole = Math.floor(value);
  const decimal = value - whole;

  if (decimal === 0) return whole;
  if (decimal <= 0.5) return whole + 1;
  return whole + 2;
};

const axisLimit = getRadarScaleMax(maxEvidence);
  const overallTone =
    overallScore >= 70
      ? "generally positive"
      : overallScore >= 50
      ? "mixed"
      : "challenging";
  const ragSummary = analysis.rag?.summary;
  const ragInsights = analysis.rag?.insights;
  const reviewEvidence = analysis.reviewEvidence.length ? analysis.reviewEvidence : remoteReviewEvidence;

  useEffect(() => {
    setRemoteReviewEvidence([]);
    if (analysis.reviewEvidence.length) {
      return;
    }

    let cancelled = false;
    void getScoredCompanyRag(analysis.companyId)
      .then((rag) => {
        if (!cancelled) {
          setRemoteReviewEvidence(parseRagEvidence(rag.evidence));
        }
      })
      .catch(() => {
        if (!cancelled) {
          setRemoteReviewEvidence([]);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [analysis.companyId, analysis.reviewEvidence.length]);

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
            {analysisDate}
          </div>
        </div>
      </header>

      <main className="container-wide section-padding">
        <section className="mb-16">
          <div className="text-center mb-8">
            <span className="inline-block text-sm font-medium text-primary uppercase tracking-wider mb-2">
              Analysis Complete
            </span>
            <h1 className="font-serif text-4xl md:text-5xl font-semibold text-foreground mb-4">
              {analysis.companyName}
            </h1>
            <p className="text-muted-foreground">
              Based on {reviewCount.toLocaleString()} employee reviews
            </p>
          </div>

          <div className="grid sm:grid-cols-3 gap-4 max-w-3xl mx-auto">
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="text-4xl font-serif font-bold text-primary mb-2">
                  {overallScore}%
                </div>
                <p className="text-sm text-muted-foreground">Overall Score</p>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  <span className="text-lg font-medium">{strongestDomain.domain}</span>
                </div>
                <p className="text-sm text-muted-foreground">Strongest Domain</p>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <TrendingDown className="w-5 h-5 text-destructive" />
                  <span className="text-lg font-medium">{weakestDomain.domain}</span>
                </div>
                <p className="text-sm text-muted-foreground">Needs Attention</p>
              </CardContent>
            </Card>
          </div>
        </section>

        <section className="mb-16">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Executive Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              {ragSummary ? (
                <div className="space-y-4">
                  {ragSummary.executiveSummary.map((paragraph, index) => (
                    <p key={index} className="text-muted-foreground leading-relaxed">
                      {paragraph}
                    </p>
                  ))}

                  {(ragSummary.keyStrengths.length > 0 || ragSummary.keyRisks.length > 0) && (
                    <div className="grid gap-4 pt-2 md:grid-cols-2">
                      {ragSummary.keyStrengths.length > 0 && (
                        <div className="rounded-lg border border-border bg-primary/5 p-4">
                          <h3 className="mb-3 text-sm font-medium text-foreground">Key strengths</h3>
                          <ul className="space-y-2 text-sm text-muted-foreground">
                            {ragSummary.keyStrengths.slice(0, 3).map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {ragSummary.keyRisks.length > 0 && (
                        <div className="rounded-lg border border-border bg-destructive/5 p-4">
                          <h3 className="mb-3 text-sm font-medium text-foreground">Key risks</h3>
                          <ul className="space-y-2 text-sm text-muted-foreground">
                            {ragSummary.keyRisks.slice(0, 3).map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-muted-foreground leading-relaxed">
                  {analysis.companyName} performs strongest in{" "}
                  <strong className="text-foreground">{strongestDomain.domain}</strong>{" "}
                  ({strongestDomain.scorePct.toFixed(1)}% normalized goal score). The weakest
                  signal appears in{" "}
                  <strong className="text-foreground">{weakestDomain.domain}</strong>{" "}
                  ({weakestDomain.scorePct.toFixed(1)}%). Across {domainScores.length} domains,
                  the company has an overall score of {overallScore}%, indicating a{" "}
                  {overallTone} employee experience profile.
                </p>
              )}
            </CardContent>
          </Card>
        </section>

        <section className="mb-16">
          <h2 className="font-serif text-2xl font-semibold text-foreground mb-6">
            Download Data
          </h2>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <DownloadCard
              title="Cleaned Reviews"
              subtitle="Pre-processed review data"
              href={analysis.downloads.cleanedReviews}
              icon={<FileSpreadsheet className="w-6 h-6" />}
            />
            <DownloadCard
              title="Review Scores"
              subtitle="Goal scores per review"
              href={analysis.downloads.reviewScores}
              icon={<BarChart3 className="w-6 h-6" />}
            />
            <DownloadCard
              title="Aggregated Scores"
              subtitle="Company-level summary"
              href={analysis.downloads.companyScores}
              icon={<FileText className="w-6 h-6" />}
            />
            <DownloadCard
              title="Topic Clusters"
              subtitle="Cluster-level language signals"
              href={analysis.downloads.topicSummary}
              icon={<BarChart3 className="w-6 h-6" />}
            />
          </div>
        </section>

        <section className="mb-16">
          <h2 className="font-serif text-2xl font-semibold text-foreground mb-6">
            Visualizations
          </h2>
          <div className="grid lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Goal Fulfillment vs. Hindrance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={barData}
                      layout="vertical"
                      margin={{ top: 5, right: 20, left: 100, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis
                        type="number"
                        domain={[-axisLimit, axisLimit]}
                        stroke="hsl(var(--muted-foreground))"
                      />
                      <YAxis
                        dataKey="domain"
                        type="category"
                        stroke="hsl(var(--muted-foreground))"
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip
                        formatter={(value: number, name: string) => {
                          if (name === "Hindrance") {
                            return [Math.abs(value).toFixed(2), name];
                          }
                          return [Number(value).toFixed(2), name];
                        }}
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Legend />
                      <Bar
                        dataKey="fulfillment"
                        fill="hsl(var(--primary))"
                        name="Fulfillment"
                        radius={[0, 4, 4, 0]}
                      />
                      <Bar
                        dataKey="hindranceNeg"
                        fill="hsl(var(--destructive))"
                        name="Hindrance"
                        radius={[4, 0, 0, 4]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

          <Card>
          <CardHeader>
            <CardTitle>Final Goal Score Profile</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[420px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart
                  cx="50%"
                  cy="48%"
                  outerRadius="82%"
                  data={radarData}
                  margin={{ top: 10, right: 30, bottom: 10, left: 30 }}
                >
                  <PolarGrid stroke="hsl(var(--border))" />
                  <PolarAngleAxis
                    dataKey="domain"
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                  />
                  <PolarRadiusAxis
                    angle={90}
                    domain={[0, 100]}
                    tickCount={5}
                    tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                  />
                  <Radar
                    name="Final Score"
                    dataKey="score"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                  />
                  <Legend />
                  <Tooltip
                    formatter={(value: number) => [`${Number(value).toFixed(1)}%`, "Final Score"]}
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
          </div>
        </section>

        {analysis.topicClusters.length > 0 && (
          <section className="mb-16">
            <div className="mb-6">
              <h2 className="font-serif text-2xl font-semibold text-foreground mb-3">
                Employee Language Clusters
              </h2>
              <p className="text-muted-foreground max-w-3xl">
                These clusters summarize recurring language patterns behind the goal scores.
                Switch between fulfillment and hindrance signals to see what employees
                most often describe.
              </p>
            </div>

            <div className="mb-5 inline-flex rounded-xl border border-border bg-card p-1">
              <button
                type="button"
                onClick={() => setClusterMode("fulfillment")}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  clusterMode === "fulfillment"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Fulfillment Signals
              </button>
              <button
                type="button"
                onClick={() => setClusterMode("hindrance")}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  clusterMode === "hindrance"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Hindrance Signals
              </button>
            </div>

            <div className="grid gap-6 lg:grid-cols-[1.2fr_1fr]">
              <Card>
                <CardHeader>
                  <CardTitle>Cluster Bubble Map</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-4 flex flex-wrap gap-2 text-xs">
                    {activeClusters.map((cluster, index) => (
                      <span
                        key={cluster.clusterId}
                        className="inline-flex items-center gap-2 rounded-full border border-border bg-background px-3 py-1.5"
                      >
                        <span
                          className="h-2.5 w-2.5 rounded-full"
                          style={{ backgroundColor: CLUSTER_COLORS[index % CLUSTER_COLORS.length] }}
                        />
                        {cluster.label}
                      </span>
                    ))}
                  </div>

                  <div className="h-[360px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={[0, 100]}
                          ticks={[20, 50, 80]}
                          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          domain={[0, 100]}
                          ticks={[25, 50, 75]}
                          tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                        />
                        <ZAxis type="number" dataKey="reviewCount" range={[180, 900]} />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          content={({ active, payload }) => {
                            if (!active || !payload?.length) return null;
                            const item = payload[0].payload as (typeof activeClusters)[number];
                            return (
                              <div className="rounded-lg border border-border bg-card p-3 text-sm shadow-card">
                                <p className="font-medium text-foreground">{item.label}</p>
                                <p className="text-muted-foreground">{item.domain}</p>
                                <p className="mt-2 text-muted-foreground">
                                  {item.reviewCount} reviews | signal {item.signal}%
                                </p>
                                <p className="mt-1 text-muted-foreground">
                                  Terms: {item.terms.slice(0, 5).join(", ")}
                                </p>
                              </div>
                            );
                          }}
                        />
                        <Scatter data={activeClusters}>
                          {activeClusters.map((cluster, index) => (
                            <Cell
                              key={cluster.clusterId}
                              fill={CLUSTER_COLORS[index % CLUSTER_COLORS.length]}
                              fillOpacity={0.72}
                            />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-4 rounded-lg border border-border bg-muted/40 p-3">
                    <p className="font-medium text-foreground">How to read this</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      Each bubble represents a recurring review theme. Larger bubbles
                      contain more reviews, higher bubbles have stronger signal, and the
                      horizontal position separates different theme families.
                    </p>
                  </div>
                </CardContent>
              </Card>

              <div className="grid gap-4">
                {ragInsights?.whatStandsOut.length ? (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2 text-lg">
                        <Sparkles className="h-4 w-4" />
                        What stands out
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3 text-sm text-muted-foreground">
                        {ragInsights.whatStandsOut.slice(0, 4).map((item) => (
                          <p key={item}>{item}</p>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                ) : null}

                {activeClusters.map((cluster) => (
                  <Card key={cluster.clusterId}>
                    <CardContent className="pt-5">
                      <div className="mb-3 flex items-start justify-between gap-3">
                        <div>
                          <h3 className="font-medium text-foreground">{cluster.label}</h3>
                          <p className="text-sm text-muted-foreground">{cluster.domain}</p>
                        </div>
                        <span className="rounded-full bg-secondary px-3 py-1 text-xs font-medium">
                          {cluster.reviewCount} reviews
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        {cluster.terms.slice(0, 5).map((term) => (
                          <span
                            key={term}
                            className="rounded-full bg-muted px-2.5 py-1 text-xs text-muted-foreground"
                          >
                            {term}
                          </span>
                        ))}
                      </div>
                      {cluster.summary ? (
                        <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
                          {cluster.summary}
                        </p>
                      ) : null}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </section>
        )}

        {reviewEvidence.length > 0 && (
          <section className="mb-16">
            <div className="mb-6">
              <h2 className="font-serif text-2xl font-semibold text-foreground mb-3">
                Review Evidence Behind the Signals
              </h2>
              <p className="text-muted-foreground max-w-3xl">
                These excerpts connect the company-level clusters back to actual employee review text.
                The highlighted context explains why a review is being treated as strength or risk
                evidence for this company, not as a personalized match judgment.
              </p>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Company review evidence
                </CardTitle>
              </CardHeader>
              <CardContent className="grid gap-4 md:grid-cols-2">
                {reviewEvidence.map((item, index) => (
                  <CompanyEvidenceCard key={`${item.label}-${item.date}-${index}`} item={item} />
                ))}
              </CardContent>
            </Card>
          </section>
        )}

        <section>
          <Card className="mb-8 bg-muted/30">
            <CardContent className="pt-6">
              <div className="grid gap-5 md:grid-cols-[1fr_auto] md:items-center">
                <div>
                  <h3 className="font-serif text-2xl font-semibold">
                    Match yourself to {analysis.companyName}
                  </h3>
                  <p className="mt-2 text-muted-foreground">
                    Answer the same workplace fit questions and see how your profile matches this company.
                  </p>
                </div>
                <Button onClick={onMatchStart} className="gap-2">
                  <Sparkles className="h-4 w-4" />
                  Show My Match
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="mb-8 bg-muted/30">
            <CardContent className="pt-6">
              <div className="grid gap-5 md:grid-cols-[1fr_auto] md:items-center">
                <div>
                  <h3 className="font-serif text-2xl font-semibold">
                    Compare {analysis.companyName} with other companies
                  </h3>
                  <p className="mt-2 text-muted-foreground">
                    Keep this company selected and add peers for a side-by-side comparison.
                  </p>
                </div>
                <Button onClick={() => onCompare?.(analysis.companyId)} className="gap-2">
                  <BarChart3 className="h-4 w-4" />
                  Compare Companies
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-muted/30">
            <CardHeader>
              <CardTitle>How to Read These Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-muted-foreground">
              <p>
                <strong className="text-foreground">Fulfillment and hindrance evidence</strong>{" "}
                are averaged from review-level goal signals and plotted per domain.
              </p>
              <p>
                <strong className="text-foreground">Important reminder:</strong> These results
                reflect aggregated employee experiences in public reviews. They show language
                patterns, not objective ground truth about policies.
              </p>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}
