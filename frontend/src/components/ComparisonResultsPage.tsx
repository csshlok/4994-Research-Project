import { useMemo, useState } from "react";
import {
  ArrowLeft,
  BarChart3,
  Download,
  FileText,
  GitCompare,
  Layers3,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  COMPARISON_DOMAINS,
  CompanyComparisonMetric,
  scoreClass,
} from "@/lib/comparison";
import { getScoredCompanyDownloadUrl, type ComparisonRagSummary } from "@/lib/backend-api";

interface ComparisonResultsPageProps {
  metrics: CompanyComparisonMetric[];
  ragSummary?: ComparisonRagSummary | null;
  onBack: () => void;
  onAddComparison: (seedCompanyId?: string) => void;
}

type Tab = "overview" | "summary" | "company";

const COLORS = ["hsl(var(--primary))", "hsl(var(--olive))", "hsl(var(--destructive))"];

function domainStats(metric: CompanyComparisonMetric) {
  const rows = COMPARISON_DOMAINS.map((domain) => ({
    key: domain.key,
    label: domain.label,
    score: metric.domains[domain.key],
  }));
  return {
    strongest: rows.reduce((a, b) => (a.score >= b.score ? a : b)),
    weakest: rows.reduce((a, b) => (a.score <= b.score ? a : b)),
  };
}

function spreadRows(metrics: CompanyComparisonMetric[]) {
  return COMPARISON_DOMAINS.map((domain) => {
    const values = metrics.map((m) => m.domains[domain.key]);
    const max = Math.max(...values);
    const min = Math.min(...values);
    const leader = metrics.find((m) => m.domains[domain.key] === max)?.label || "";
    const trailer = metrics.find((m) => m.domains[domain.key] === min)?.label || "";
    return {
      domain: domain.label,
      spread: max - min,
      insight: `${leader} leads ${trailer} by ${max - min} points.`,
    };
  })
    .sort((a, b) => b.spread - a.spread)
    .slice(0, 3);
}

function relativeHeatmapStyle(metrics: CompanyComparisonMetric[], domainKey: string, score: number) {
  const values = metrics.map((metric) => metric.domains[domainKey]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const ratio = max === min ? 0.5 : (score - min) / (max - min);
  const hue = ratio < 0.5 ? 0 : 112;
  const saturation = ratio < 0.5 ? 58 - ratio * 22 : 34 + ratio * 12;
  const lightness = ratio < 0.5 ? 91 - ratio * 3 : 88 - ratio * 10;
  return {
    backgroundColor: `hsl(${hue} ${saturation}% ${lightness}%)`,
    color: ratio >= 0.5 ? "hsl(100 35% 18%)" : "hsl(0 70% 42%)",
  };
}

export function ComparisonResultsPage({
  metrics,
  ragSummary,
  onBack,
  onAddComparison,
}: ComparisonResultsPageProps) {
  const [tab, setTab] = useState<Tab>("overview");
  const [selectedId, setSelectedId] = useState(metrics[0]?.id || "");

  const selected = metrics.find((m) => m.id === selectedId) || metrics[0];
  const selectedStats = selected ? domainStats(selected) : null;
  const orderedMetrics = useMemo(() => {
    if (!selected) return metrics;
    return [selected, ...metrics.filter((m) => m.id !== selected.id)];
  }, [metrics, selected]);

  const selectedRadarRows = selected
    ? COMPARISON_DOMAINS.map((domain) => {
        const peers = metrics.filter((m) => m.id !== selected.id);
        const peerAverage = Math.round(
          peers.reduce((sum, m) => sum + m.domains[domain.key], 0) / Math.max(1, peers.length)
        );
        return { domain: domain.label, selected: selected.domains[domain.key], peerAverage };
      })
    : [];

  const clusterRows = metrics.flatMap((metric) =>
    metric.topics
      .filter((topic) => topic.mode === "hindrance")
      .sort((a, b) => b.reviewCount - a.reviewCount)
      .slice(0, 2)
      .map((topic) => ({ ...topic, company: metric.label, companyId: metric.id }))
  );

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
        <div className="container-wide flex items-center justify-between py-4">
          <Button variant="ghost" onClick={onBack} className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            New Analysis
          </Button>
          <span aria-hidden="true" />
        </div>
      </header>

      <main className="container-wide py-10 md:py-14">
        <section className="mb-10">
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
            Company Comparison
          </p>
          <h1 className="font-serif text-4xl font-semibold text-foreground md:text-5xl">
            Multi-Company Dashboard
          </h1>
          <p className="mt-4 max-w-3xl text-lg leading-relaxed text-muted-foreground">
            Compare precomputed company scores, domain profiles, and recurring language clusters.
          </p>
        </section>

        <div className="mb-6 grid gap-4 md:grid-cols-3">
          {orderedMetrics.map((metric) => {
            const isSelected = selected?.id === metric.id;
            const stats = domainStats(metric);
            return (
              <Card
                key={metric.id}
                onClick={() => {
                  setSelectedId(metric.id);
                  setTab("company");
                }}
                className={`cursor-pointer transition-all ${
                  isSelected
                    ? "border-primary shadow-elevated"
                    : "hover:border-primary/40"
                }`}
              >
                <CardContent className="pt-6">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="font-medium">{metric.label}</h3>
                    <Badge className={scoreClass(metric.overall)}>{metric.overall}%</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {metric.reviews.toLocaleString()} scored reviews
                  </p>
                  {isSelected ? (
                    <div className="mt-5 grid gap-3 text-sm">
                      <div className="flex items-center justify-between rounded-lg bg-primary/10 px-3 py-2">
                        <span className="inline-flex items-center gap-2 text-primary">
                          <TrendingUp className="h-4 w-4" />
                          Strongest
                        </span>
                        <span className="font-medium">{stats.strongest.label}</span>
                      </div>
                      <div className="flex items-center justify-between rounded-lg bg-destructive/10 px-3 py-2">
                        <span className="inline-flex items-center gap-2 text-destructive">
                          <TrendingDown className="h-4 w-4" />
                          Weakest
                        </span>
                        <span className="font-medium">{stats.weakest.label}</span>
                      </div>
                    </div>
                  ) : (
                    <p className="mt-4 text-xs text-muted-foreground">
                      Click to center this company and inspect its details.
                    </p>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>

        <div className="mb-6 flex flex-wrap gap-3">
          <div className="inline-flex rounded-xl border border-border bg-card p-1">
            {[
              { key: "overview", label: "Overview" },
              { key: "summary", label: "Executive Summary" },
              { key: "company", label: "Company Detail" },
            ].map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setTab(item.key as Tab)}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  tab === item.key
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <Button variant="outline" onClick={() => onAddComparison()} className="gap-2">
            <GitCompare className="h-4 w-4" />
            Compare More Companies
          </Button>
        </div>

        {tab === "summary" ? (
          <div className="grid gap-6 lg:grid-cols-[1fr_0.8fr]">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Executive Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4 text-muted-foreground">
                {ragSummary?.executive_summary?.length ? (
                  <>
                    {ragSummary.executive_summary.map((paragraph, index) => (
                      <p key={index}>{paragraph}</p>
                    ))}
                    {ragSummary.source === "fallback" ? (
                      <p className="text-xs text-muted-foreground">
                        Gemini was unavailable for this request, so this summary uses cached score gaps and company RAG profiles.
                      </p>
                    ) : null}
                  </>
                ) : (
                  <>
                    <p>
                      This comparison is generated from cached company metrics and RAG
                      profiles. It uses precomputed score differences, cluster signals,
                      and single-company summaries to explain the selected companies.
                    </p>
                    <p>
                      Use the selected company card to focus the radar and detail view.
                      The strongest and weakest domains show which needs are most
                      consistently fulfilled or hindered for that company relative to peers.
                    </p>
                  </>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Key Takeaways</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                {(ragSummary?.key_differences?.length ? ragSummary.key_differences : spreadRows(metrics).map((row) => row.insight)).map((insight) => (
                  <div key={insight} className="rounded-lg border border-border p-3">
                    <p className="text-muted-foreground">{insight}</p>
                  </div>
                ))}
              </CardContent>
            </Card>

            {ragSummary?.best_fit_by_need?.length ? (
              <Card>
                <CardHeader>
                  <CardTitle>Best Fit by Need</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  {ragSummary.best_fit_by_need.map((item) => (
                    <div key={`${item.need}-${item.company}`} className="rounded-lg border border-border p-3">
                      <div className="mb-1 flex items-center justify-between gap-3">
                        <p className="font-medium">{item.need}</p>
                        <Badge variant="secondary">{item.company}</Badge>
                      </div>
                      <p className="text-muted-foreground">{item.reason}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            ) : null}

            {ragSummary?.company_notes?.length ? (
              <Card>
                <CardHeader>
                  <CardTitle>Company Notes</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  {ragSummary.company_notes.map((item) => (
                    <div key={item.company} className="rounded-lg border border-border p-3">
                      <p className="mb-2 font-medium">{item.company}</p>
                      <p className="text-muted-foreground">
                        <span className="text-primary">Strength:</span> {item.strength}
                      </p>
                      <p className="mt-1 text-muted-foreground">
                        <span className="text-destructive">Risk:</span> {item.risk}
                      </p>
                    </div>
                  ))}
                </CardContent>
              </Card>
            ) : null}
          </div>
        ) : tab === "company" && selected && selectedStats ? (
          <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
            <Card>
              <CardHeader>
                <CardTitle>{selected.label} Detail</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-6 grid grid-cols-2 gap-3">
                  <div className="rounded-lg bg-muted p-4">
                    <p className="text-sm text-muted-foreground">Overall Score</p>
                    <p className="mt-1 text-3xl font-semibold text-primary">{selected.overall}%</p>
                  </div>
                  <div className="rounded-lg bg-muted p-4">
                    <p className="text-sm text-muted-foreground">Scored Reviews</p>
                    <p className="mt-1 text-3xl font-semibold">{selected.reviews.toLocaleString()}</p>
                  </div>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between rounded-lg border border-border p-3">
                    <span className="inline-flex items-center gap-2 text-primary">
                      <TrendingUp className="h-4 w-4" />
                      Strongest Domain
                    </span>
                    <span className="font-medium">{selectedStats.strongest.label}</span>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-border p-3">
                    <span className="inline-flex items-center gap-2 text-destructive">
                      <TrendingDown className="h-4 w-4" />
                      Weakest Domain
                    </span>
                    <span className="font-medium">{selectedStats.weakest.label}</span>
                  </div>
                </div>
                <div className="mt-6 grid gap-3 sm:grid-cols-2">
                  <Button variant="outline" asChild className="gap-2">
                    <a href={getScoredCompanyDownloadUrl(selected.id, "review_scores.csv")} target="_blank" rel="noreferrer">
                      <Download className="h-4 w-4" />
                      Download Reviews
                    </a>
                  </Button>
                  <Button variant="outline" asChild className="gap-2">
                    <a href={getScoredCompanyDownloadUrl(selected.id, "company_scores.csv")} target="_blank" rel="noreferrer">
                      <BarChart3 className="h-4 w-4" />
                      Download Scores
                    </a>
                  </Button>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>{selected.label} vs Peer Average</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={selectedRadarRows} outerRadius="74%">
                      <PolarGrid stroke="hsl(var(--border))" />
                      <PolarAngleAxis dataKey="domain" tick={{ fontSize: 11 }} />
                      <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Radar name={selected.label} dataKey="selected" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.28} />
                      <Radar name="Peer average" dataKey="peerAverage" stroke="hsl(var(--olive))" fill="hsl(var(--olive))" fillOpacity={0.12} />
                      <Legend />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Domain Gap Analysis
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {spreadRows(metrics).map((row) => (
                  <div key={row.domain} className="rounded-lg border border-border p-4">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <h3 className="font-medium">{row.domain}</h3>
                      <Badge variant="secondary">{row.spread} point spread</Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{row.insight}</p>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Final Goal Score Profile</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Selected company compared with the average of the other selected companies.
                </p>
              </CardHeader>
              <CardContent>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={selectedRadarRows} outerRadius="72%">
                      <PolarGrid stroke="hsl(var(--border))" />
                      <PolarAngleAxis dataKey="domain" tick={{ fontSize: 11 }} />
                      <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Radar
                        name={selected?.label || "Selected"}
                        dataKey="selected"
                        stroke="hsl(var(--primary))"
                        fill="hsl(var(--primary))"
                        fillOpacity={0.26}
                        strokeWidth={2}
                      />
                      <Radar
                        name="Peer average"
                        dataKey="peerAverage"
                        stroke="hsl(var(--olive))"
                        fill="hsl(var(--olive))"
                        fillOpacity={0.12}
                        strokeWidth={2}
                      />
                      <Legend />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Company x Goal Heatmap</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[760px] border-separate border-spacing-3 text-sm">
                    <thead>
                      <tr>
                        <th className="text-left text-base font-medium text-muted-foreground">Company</th>
                        {COMPARISON_DOMAINS.map((domain) => (
                          <th key={domain.key} className="text-center text-base font-medium text-muted-foreground">
                            {domain.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.map((metric) => (
                        <tr key={metric.id}>
                          <td className="py-3 text-lg font-medium">{metric.label}</td>
                          {COMPARISON_DOMAINS.map((domain) => (
                            <td
                              key={domain.key}
                              className="rounded-lg px-4 py-5 text-center text-lg font-semibold"
                              style={relativeHeatmapStyle(metrics, domain.key, metric.domains[domain.key])}
                            >
                              {metric.domains[domain.key]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Layers3 className="h-5 w-5" />
                  Shared Cluster Bubble Map
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
                  <div>
                    <div className="mb-4 flex flex-wrap gap-2 text-xs">
                      {metrics.map((metric, i) => (
                        <span key={metric.id} className="inline-flex items-center gap-2 rounded-full border border-border bg-background px-2.5 py-1">
                          <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                          {metric.label}
                        </span>
                      ))}
                    </div>
                    <div className="h-[440px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 26, right: 26, bottom: 24, left: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                          <XAxis type="number" dataKey="x" domain={[0, 100]} ticks={[20, 50, 80]} />
                          <YAxis type="number" dataKey="y" domain={[0, 100]} ticks={[25, 50, 75]} />
                          <ZAxis type="number" dataKey="reviewCount" range={[220, 980]} />
                          <Tooltip
                            content={({ active, payload }) => {
                              if (!active || !payload?.length) return null;
                              const item = payload[0].payload as (typeof clusterRows)[number];
                              return (
                                <div className="rounded-lg border border-border bg-card p-3 text-sm shadow-card">
                                  <p className="font-medium">{item.label}</p>
                                  <p className="text-muted-foreground">{item.company} | {item.domain}</p>
                                  <p className="text-muted-foreground">{item.reviewCount} reviews</p>
                                </div>
                              );
                            }}
                          />
                          {metrics.map((metric, i) => (
                            <Scatter
                              key={metric.id}
                              name={metric.label}
                              data={clusterRows.filter((row) => row.companyId === metric.id)}
                              fill={COLORS[i % COLORS.length]}
                              fillOpacity={0.72}
                            >
                              {clusterRows.filter((row) => row.companyId === metric.id).map((row) => (
                                <Cell key={`${row.company}-${row.label}`} fill={COLORS[i % COLORS.length]} />
                              ))}
                            </Scatter>
                          ))}
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="rounded-lg bg-muted p-4">
                      <p className="font-medium text-foreground">How to read it</p>
                      <div className="mt-3 grid gap-2 text-sm text-muted-foreground">
                        <p>Higher bubbles indicate stronger employee language signals.</p>
                        <p>Larger bubbles represent more reviews in that theme.</p>
                        <p>Color identifies which company owns the cluster.</p>
                      </div>
                    </div>
                    <div className="rounded-lg border border-border p-4">
                      <p className="font-medium text-foreground">What stands out</p>
                      <p className="mt-2 text-sm text-muted-foreground">
                        The map highlights the largest hindrance clusters for each selected company.
                      </p>
                    </div>
                    <div className="rounded-lg border border-border p-4">
                      <p className="font-medium text-foreground">Cluster terms</p>
                      <div className="mt-3 space-y-3">
                        {clusterRows.slice(0, 4).map((cluster) => (
                          <div key={`${cluster.company}-${cluster.label}`}>
                            <p className="text-sm font-medium">{cluster.label}</p>
                            <p className="text-xs text-muted-foreground">
                              {cluster.company}: {cluster.terms.slice(0, 4).join(", ")}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
