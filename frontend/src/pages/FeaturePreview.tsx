import { useMemo, useState } from "react";
import {
  ArrowLeft,
  BarChart3,
  Check,
  Download,
  FileText,
  GitCompare,
  Layers3,
  LineChart,
  Plus,
  Search,
  TrendingDown,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  LabelList,
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
import { Link } from "react-router-dom";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { COMPANY_OPTIONS } from "@/lib/company-options";

type ClusterMode = "fulfillment" | "hindrance";
type DashboardTab = "overview" | "summary" | "company";

const clusterData: Record<
  ClusterMode,
  Array<{
    id: string;
    label: string;
    domain: string;
    terms: string[];
    reviews: number;
    sentiment: number;
    x: number;
    y: number;
  }>
> = {
  fulfillment: [
    {
      id: "culture",
      label: "Culture & Belonging",
      domain: "Affiliation",
      terms: ["supportive", "friendly", "teamwork", "collaboration", "coworkers"],
      reviews: 218,
      sentiment: 82,
      x: 22,
      y: 74,
    },
    {
      id: "growth",
      label: "Growth & Recognition",
      domain: "Status & Esteem",
      terms: ["promotion", "learning", "mentorship", "visibility", "advancement"],
      reviews: 164,
      sentiment: 76,
      x: 48,
      y: 58,
    },
    {
      id: "flex",
      label: "Flexibility & Care",
      domain: "Family Care",
      terms: ["remote work", "pto", "flexible hours", "family time", "balance"],
      reviews: 136,
      sentiment: 73,
      x: 68,
      y: 79,
    },
    {
      id: "benefits",
      label: "Pay & Benefits",
      domain: "Physiological",
      terms: ["benefits", "salary", "bonus", "healthcare", "401k"],
      reviews: 188,
      sentiment: 70,
      x: 82,
      y: 45,
    },
  ],
  hindrance: [
    {
      id: "burnout",
      label: "Burnout & Workload",
      domain: "Physiological",
      terms: ["long hours", "burnout", "understaffed", "overtime", "stress"],
      reviews: 246,
      sentiment: 24,
      x: 24,
      y: 28,
    },
    {
      id: "fairness",
      label: "Unsafe or Unfair Culture",
      domain: "Self-Protection",
      terms: ["toxic", "favoritism", "retaliation", "harassment", "discrimination"],
      reviews: 119,
      sentiment: 18,
      x: 50,
      y: 38,
    },
    {
      id: "stagnation",
      label: "Stagnation",
      domain: "Status & Esteem",
      terms: ["no promotion", "no feedback", "ignored", "dead end", "limited growth"],
      reviews: 151,
      sentiment: 31,
      x: 70,
      y: 24,
    },
    {
      id: "balance",
      label: "Work-Life Pressure",
      domain: "Family Care",
      terms: ["always on", "late nights", "weekend work", "no flexibility", "travel"],
      reviews: 176,
      sentiment: 27,
      x: 82,
      y: 52,
    },
  ],
};

const selectedCompanies = ["Microsoft", "Adobe", "Google"];

const comparisonRows = [
  { company: "Microsoft", overall: 76, reviews: 940, phys: 72, safety: 69, aff: 82, status: 78, family: 73 },
  { company: "Adobe", overall: 81, reviews: 610, phys: 78, safety: 76, aff: 85, status: 80, family: 77 },
  { company: "Google", overall: 74, reviews: 1120, phys: 83, safety: 65, aff: 76, status: 75, family: 61 },
];

const topicComparisonRows = [
  { topic: "Culture", Microsoft: 34, Adobe: 39, Google: 28 },
  { topic: "Growth", Microsoft: 29, Adobe: 25, Google: 31 },
  { topic: "Benefits", Microsoft: 23, Adobe: 21, Google: 36 },
  { topic: "Workload", Microsoft: 18, Adobe: 14, Google: 33 },
  { topic: "Flexibility", Microsoft: 27, Adobe: 31, Google: 19 },
];

const domainGapRows = [
  {
    domain: "Family Care",
    insight: "Google trails the peer average most clearly on work-life support.",
    spread: 16,
  },
  {
    domain: "Self-Protection",
    insight: "Adobe leads the comparison on safety and fairness language.",
    spread: 11,
  },
  {
    domain: "Affiliation",
    insight: "All three companies show relatively strong belonging signals.",
    spread: 9,
  },
];

const balanceRows = [
  { domain: "Physiological", fulfillment: 72, hindrance: 28 },
  { domain: "Self-Protection", fulfillment: 58, hindrance: 42 },
  { domain: "Affiliation", fulfillment: 81, hindrance: 19 },
  { domain: "Status & Esteem", fulfillment: 74, hindrance: 26 },
  { domain: "Family Care", fulfillment: 54, hindrance: 46 },
];

const evidenceSnapshot = {
  strongest: ["supportive coworkers", "collaborative teams", "friendly culture"],
  weakest: ["limited flexibility", "late meetings", "after-hours pressure"],
};

const companyClusterRows = [
  {
    company: "Microsoft",
    label: "Team Culture",
    domain: "Affiliation",
    terms: ["teamwork", "supportive", "collaboration"],
    reviews: 214,
    intensity: 82,
    x: 24,
    y: 76,
    fill: "hsl(var(--primary))",
  },
  {
    company: "Microsoft",
    label: "Career Growth",
    domain: "Status & Esteem",
    terms: ["learning", "promotion", "mentorship"],
    reviews: 148,
    intensity: 73,
    x: 42,
    y: 63,
    fill: "hsl(var(--primary))",
  },
  {
    company: "Adobe",
    label: "Balanced Culture",
    domain: "Affiliation",
    terms: ["friendly", "belonging", "respect"],
    reviews: 186,
    intensity: 85,
    x: 58,
    y: 80,
    fill: "hsl(var(--olive))",
  },
  {
    company: "Adobe",
    label: "Flexibility",
    domain: "Family Care",
    terms: ["remote work", "pto", "flexible hours"],
    reviews: 132,
    intensity: 77,
    x: 74,
    y: 66,
    fill: "hsl(var(--olive))",
  },
  {
    company: "Google",
    label: "Benefits",
    domain: "Physiological",
    terms: ["salary", "benefits", "healthcare"],
    reviews: 238,
    intensity: 83,
    x: 78,
    y: 42,
    fill: "hsl(var(--destructive))",
  },
  {
    company: "Google",
    label: "Workload Pressure",
    domain: "Family Care",
    terms: ["long hours", "burnout", "always on"],
    reviews: 251,
    intensity: 61,
    x: 36,
    y: 28,
    fill: "hsl(var(--destructive))",
  },
];

const companyClusterGroups = ["Microsoft", "Adobe", "Google"].map((company) => ({
  company,
  fill: companyClusterRows.find((row) => row.company === company)?.fill || "hsl(var(--primary))",
  rows: companyClusterRows.filter((row) => row.company === company),
}));

const heatmapDomains = [
  { key: "phys", label: "Phys", full: "Physiological" },
  { key: "safety", label: "Safety" },
  { key: "aff", label: "Affil", full: "Affiliation" },
  { key: "status", label: "Status", full: "Status & Esteem" },
  { key: "family", label: "Family", full: "Family Care" },
] as const;

type CompanyRow = (typeof comparisonRows)[number];
type DomainKey = (typeof heatmapDomains)[number]["key"];

function getDomainStats(row: CompanyRow) {
  const domains = heatmapDomains.map((domain) => ({
    key: domain.key,
    label: domain.full || domain.label,
    short: domain.label,
    score: Number(row[domain.key]),
  }));
  const strongest = domains.reduce((best, cur) => (cur.score > best.score ? cur : best));
  const weakest = domains.reduce((worst, cur) => (cur.score < worst.score ? cur : worst));
  return { domains, strongest, weakest };
}

function peerAverage(domain: DomainKey, selectedCompany: string) {
  const peers = comparisonRows.filter((row) => row.company !== selectedCompany);
  const total = peers.reduce((sum, row) => sum + Number(row[domain]), 0);
  return Math.round(total / Math.max(1, peers.length));
}

function scoreColor(score: number) {
  if (score >= 80) return "bg-primary text-primary-foreground";
  if (score >= 70) return "bg-olive-muted/70 text-foreground";
  return "bg-destructive/15 text-destructive";
}

function CompanyPill({ label, selected = false }: { label: string; selected?: boolean }) {
  return (
    <button
      type="button"
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm transition-colors ${
        selected
          ? "border-primary bg-primary text-primary-foreground"
          : "border-border bg-card hover:border-primary/40"
      }`}
    >
      {selected ? <Check className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
      {label}
    </button>
  );
}

function SectionHeader({
  eyebrow,
  title,
  body,
}: {
  eyebrow: string;
  title: string;
  body?: string;
}) {
  return (
    <div className="mb-6">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
        {eyebrow}
      </p>
      <h2 className="font-serif text-2xl font-semibold text-foreground md:text-3xl">
        {title}
      </h2>
      {body ? <p className="mt-3 max-w-3xl text-muted-foreground">{body}</p> : null}
    </div>
  );
}

export default function FeaturePreview() {
  const [clusterMode, setClusterMode] = useState<ClusterMode>("fulfillment");
  const [dashboardTab, setDashboardTab] = useState<DashboardTab>("overview");
  const [selectedCompany, setSelectedCompany] = useState("Microsoft");
  const activeClusters = clusterData[clusterMode];

  const mockCompanyChoices = useMemo(
    () => COMPANY_OPTIONS.slice(0, 10).map((company) => company.label),
    []
  );

  const selectedRow =
    comparisonRows.find((row) => row.company === selectedCompany) || comparisonRows[0];
  const selectedStats = getDomainStats(selectedRow);
  const centeredCompanyRows = useMemo(() => {
    const others = comparisonRows.filter((row) => row.company !== selectedRow.company);
    return [others[0], selectedRow, others[1]].filter(Boolean) as CompanyRow[];
  }, [selectedRow]);
  const selectedRadarRows = selectedStats.domains.map((domain) => ({
    domain: domain.short,
    selected: domain.score,
    peerAverage: peerAverage(domain.key, selectedRow.company),
  }));

  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-40 border-b border-border bg-background/90 backdrop-blur">
        <div className="container-wide flex items-center justify-between py-4">
          <Button variant="ghost" asChild className="gap-2">
            <Link to="/">
              <ArrowLeft className="h-4 w-4" />
              Back to App
            </Link>
          </Button>
          <Badge variant="outline">Sandbox Preview</Badge>
        </div>
      </header>

      <main className="container-wide py-10 md:py-14">
        <section className="mb-12">
          <div className="max-w-4xl">
            <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
              Proposed Features
            </p>
            <h1 className="font-serif text-4xl font-semibold text-foreground md:text-5xl">
              Topic clusters and company comparison preview
            </h1>
            <p className="mt-4 text-lg leading-relaxed text-muted-foreground">
              This page uses mock data to preview the interaction model before the
              production pipeline and backend endpoints are added.
            </p>
          </div>
        </section>

        <section className="mb-16">
          <SectionHeader
            eyebrow="Topic Clustering"
            title="Employee Language Clusters"
            body="Users can switch between language associated with goal fulfillment and language associated with goal hindrance."
          />

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
                <CardTitle className="flex items-center gap-2">
                  <Layers3 className="h-5 w-5" />
                  Cluster Bubble Map
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-4 flex flex-wrap gap-2 text-xs">
                  {activeClusters.map((cluster, index) => (
                    <span
                      key={cluster.id}
                      className="inline-flex items-center gap-2 rounded-full border border-border bg-background px-3 py-1.5"
                    >
                      <span
                        className="h-2.5 w-2.5 rounded-full"
                        style={{
                          backgroundColor:
                            index % 4 === 0
                              ? "hsl(var(--primary))"
                              : index % 4 === 1
                              ? "hsl(var(--olive))"
                              : index % 4 === 2
                              ? "hsl(var(--destructive))"
                              : "hsl(var(--forest-light))",
                        }}
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
                        name="Theme separation"
                        domain={[0, 100]}
                        ticks={[20, 50, 80]}
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        name="Sentiment intensity"
                        domain={[0, 100]}
                        ticks={[25, 50, 75]}
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }}
                      />
                      <ZAxis type="number" dataKey="reviews" range={[220, 950]} />
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
                                {item.reviews} reviews | {item.sentiment}% sentiment
                              </p>
                            </div>
                          );
                        }}
                      />
                      <Scatter data={activeClusters} fill="hsl(var(--primary))">
                        {activeClusters.map((entry, index) => (
                          <Cell
                            key={entry.id}
                            fill={
                              index % 4 === 0
                                ? "hsl(var(--primary))"
                                : index % 4 === 1
                                ? "hsl(var(--olive))"
                                : index % 4 === 2
                                ? "hsl(var(--destructive))"
                                : "hsl(var(--forest-light))"
                            }
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
                    contain more reviews, higher bubbles have stronger sentiment signal,
                    and the horizontal position separates different theme families.
                  </p>
                </div>
              </CardContent>
            </Card>

            <div className="grid gap-4">
              {activeClusters.map((cluster) => (
                <Card key={cluster.id}>
                  <CardContent className="pt-5">
                    <div className="mb-3 flex items-start justify-between gap-3">
                      <div>
                        <h3 className="font-medium text-foreground">{cluster.label}</h3>
                        <p className="text-sm text-muted-foreground">{cluster.domain}</p>
                      </div>
                      <Badge variant="secondary">{cluster.reviews} reviews</Badge>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {cluster.terms.map((term) => (
                        <span
                          key={term}
                          className="rounded-full bg-muted px-2.5 py-1 text-xs text-muted-foreground"
                        >
                          {term}
                        </span>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        <section className="mb-16">
          <SectionHeader
            eyebrow="Comparison Entry"
            title="Landing Page Comparison Selector"
            body="This is how the first-page entry could let users choose several companies before running a comparison."
          />
          <Card>
            <CardContent className="pt-6">
              <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
                <div>
                  <div className="mb-4 inline-flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-sm font-medium text-primary">
                    <GitCompare className="h-4 w-4" />
                    Compare Companies
                  </div>
                  <h3 className="mb-3 font-serif text-2xl font-semibold">
                    Select companies from the scored review database
                  </h3>
                  <p className="text-muted-foreground">
                    The production version would read from pre-scored company folders,
                    making this flow fast enough for hosted use on limited CPU.
                  </p>
                </div>
                <div>
                  <div className="mb-4 flex items-center gap-2 rounded-xl border border-border bg-background px-3 py-2">
                    <Search className="h-4 w-4 text-primary" />
                    <span className="text-sm text-muted-foreground">Search companies...</span>
                  </div>
                  <div className="mb-5 flex flex-wrap gap-2">
                    {mockCompanyChoices.map((company) => (
                      <CompanyPill
                        key={company}
                        label={company}
                        selected={selectedCompanies.includes(company)}
                      />
                    ))}
                  </div>
                  <Button className="gap-2">
                    Run Comparison
                    <BarChart3 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="mb-16">
          <SectionHeader
            eyebrow="Results Page Entry"
            title="Compare From A Single Company Result"
            body="After viewing one company, users can add comparison companies without starting over."
          />
          <Card className="bg-muted/30">
            <CardContent className="pt-6">
              <div className="grid gap-5 md:grid-cols-[1fr_auto] md:items-center">
                <div>
                  <h3 className="font-serif text-2xl font-semibold">
                    Compare Microsoft with other companies
                  </h3>
                  <p className="mt-2 text-muted-foreground">
                    Microsoft stays selected, and the user adds companies for a
                    side-by-side comparison.
                  </p>
                </div>
                <Button className="gap-2">
                  Compare Companies
                  <GitCompare className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </section>

        <section>
          <SectionHeader
            eyebrow="Comparison Dashboard"
            title="Multi-Company Comparison"
            body="The dashboard keeps the current palette but becomes more data dense, with score, domain, and topic views."
          />

          <div className="mb-6 grid gap-4 md:grid-cols-3">
            {centeredCompanyRows.map((row, index) => {
              const isSelected = row.company === selectedRow.company;
              const stats = getDomainStats(row);
              return (
              <Card
                key={row.company}
                className={`cursor-pointer transition-all ${
                  isSelected
                    ? "order-2 border-primary shadow-elevated md:scale-[1.03]"
                    : index === 0
                    ? "order-1 hover:border-primary/40"
                    : "order-3 hover:border-primary/40"
                }`}
                onClick={() => {
                  setSelectedCompany(row.company);
                  setDashboardTab("company");
                }}
              >
                <CardContent className="pt-6">
                  <div className="mb-4 flex items-center justify-between">
                    <h3 className="font-medium">{row.company}</h3>
                    <Badge className={scoreColor(row.overall)}>{row.overall}%</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {row.reviews.toLocaleString()} scored reviews
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

          <div className="mb-6 inline-flex rounded-xl border border-border bg-card p-1">
            {[
              { key: "overview", label: "Overview" },
              { key: "summary", label: "Executive Summary" },
              { key: "company", label: "Company Detail" },
            ].map((tab) => (
              <button
                key={tab.key}
                type="button"
                onClick={() => setDashboardTab(tab.key as DashboardTab)}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  dashboardTab === tab.key
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          {dashboardTab === "summary" ? (
            <div className="grid gap-6 lg:grid-cols-[1fr_0.8fr]">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    Executive Summary
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 text-muted-foreground">
                  <p>
                    Adobe leads this comparison with the strongest overall employee
                    experience score, driven by high affiliation and steady work-life
                    support signals. Microsoft follows closely, with especially strong
                    belonging and recognition language, while Google shows the strongest
                    basic compensation and benefits signal but weaker family-care and
                    safety language.
                  </p>
                  <p>
                    The main difference across the companies is not whether employees
                    describe positive experiences, but which needs are most consistently
                    supported. Adobe appears more balanced across domains, Microsoft
                    stands out for team and growth language, and Google shows more
                    polarization between benefits-related fulfillment and workload-related
                    hindrance.
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>Key Takeaways</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3 text-sm">
                  <div className="rounded-lg bg-primary/10 p-3">
                    <p className="font-medium text-primary">Most balanced profile</p>
                    <p className="text-muted-foreground">Adobe has the most even scores across all five domains.</p>
                  </div>
                  <div className="rounded-lg bg-muted p-3">
                    <p className="font-medium">Largest gap</p>
                    <p className="text-muted-foreground">Google has a wide spread between physiological and family-care scores.</p>
                  </div>
                  <div className="rounded-lg bg-destructive/10 p-3">
                    <p className="font-medium text-destructive">Common risk area</p>
                    <p className="text-muted-foreground">Workload pressure appears as the most important shared hindrance topic.</p>
                  </div>
                </CardContent>
              </Card>
            </div>
          ) : dashboardTab === "company" ? (
            <div className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
              <Card>
                <CardHeader>
                  <CardTitle>{selectedRow.company} Detail</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="mb-6 grid grid-cols-2 gap-3">
                    <div className="rounded-lg bg-muted p-4">
                      <p className="text-sm text-muted-foreground">Overall Score</p>
                      <p className="mt-1 text-3xl font-semibold text-primary">{selectedRow.overall}%</p>
                    </div>
                    <div className="rounded-lg bg-muted p-4">
                      <p className="text-sm text-muted-foreground">Scored Reviews</p>
                      <p className="mt-1 text-3xl font-semibold">{selectedRow.reviews.toLocaleString()}</p>
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
                    <Button variant="outline" className="gap-2">
                      <Download className="h-4 w-4" />
                      Download Reviews
                    </Button>
                    <Button variant="outline" className="gap-2">
                      <FileText className="h-4 w-4" />
                      Download Scores
                    </Button>
                  </div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle>{selectedRow.company} Fulfillment vs Hindrance</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={balanceRows} layout="vertical" margin={{ top: 5, right: 18, left: 92, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis type="number" domain={[0, 100]} stroke="hsl(var(--muted-foreground))" />
                        <YAxis dataKey="domain" type="category" stroke="hsl(var(--muted-foreground))" width={112} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="fulfillment" name="Fulfillment" stackId="a" fill="hsl(var(--primary))" radius={[4, 0, 0, 4]} />
                        <Bar dataKey="hindrance" name="Hindrance" stackId="a" fill="hsl(var(--destructive))" radius={[0, 4, 4, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="mt-5 grid gap-3 sm:grid-cols-2">
                    <div className="rounded-lg bg-primary/10 p-3">
                      <p className="mb-2 font-medium text-primary">Strongest evidence</p>
                      <div className="flex flex-wrap gap-2">
                        {evidenceSnapshot.strongest.map((item) => (
                          <span key={item} className="rounded-full bg-background px-2.5 py-1 text-xs">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="rounded-lg bg-destructive/10 p-3">
                      <p className="mb-2 font-medium text-destructive">Friction evidence</p>
                      <div className="flex flex-wrap gap-2">
                        {evidenceSnapshot.weakest.map((item) => (
                          <span key={item} className="rounded-full bg-background px-2.5 py-1 text-xs">
                            {item}
                          </span>
                        ))}
                      </div>
                    </div>
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
              <CardContent>
                <div className="space-y-4">
                  {domainGapRows.map((row) => (
                    <div key={row.domain} className="rounded-lg border border-border p-4">
                      <div className="mb-2 flex items-center justify-between gap-3">
                        <h3 className="font-medium">{row.domain}</h3>
                        <Badge variant="secondary">{row.spread} point spread</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{row.insight}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <LineChart className="h-5 w-5" />
                  {selectedRow.company} vs Peer Average
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-72">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={selectedRadarRows} outerRadius="74%">
                      <PolarGrid stroke="hsl(var(--border))" />
                      <PolarAngleAxis dataKey="domain" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <PolarRadiusAxis domain={[50, 100]} tick={{ fontSize: 10 }} />
                      <Radar
                        name={selectedRow.company}
                        dataKey="selected"
                        stroke="hsl(var(--primary))"
                        fill="hsl(var(--primary))"
                        fillOpacity={0.28}
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
                <p className="text-sm text-muted-foreground">
                  Wider comparison view across all selected companies and goal domains.
                </p>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[760px] border-separate border-spacing-3 text-sm">
                    <thead>
                      <tr>
                        <th className="text-left text-base font-medium text-muted-foreground">Company</th>
                        {heatmapDomains.map((domain) => (
                          <th key={domain.key} className="text-center text-base font-medium text-muted-foreground">
                            {domain.full || domain.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonRows.map((row) => (
                        <tr key={row.company}>
                          <td className="py-3 text-lg font-medium">{row.company}</td>
                          {heatmapDomains.map((domain) => {
                            const score = Number(row[domain.key]);
                            return (
                              <td key={domain.key} className={`rounded-lg px-4 py-5 text-center text-lg font-semibold ${scoreColor(score)}`}>
                                {score}
                              </td>
                            );
                          })}
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
                <p className="text-sm text-muted-foreground">
                  A larger theme map with the chart separated from the explanation panel.
                </p>
              </CardHeader>
              <CardContent>
                <div className="grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
                  <div>
                    <div className="mb-4 flex flex-wrap gap-2 text-xs">
                      {companyClusterGroups.map((group) => (
                        <span
                          key={group.company}
                          className="inline-flex items-center gap-2 rounded-full border border-border bg-background px-2.5 py-1"
                        >
                          <span
                            className="h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: group.fill }}
                          />
                          {group.company}
                        </span>
                      ))}
                    </div>
                    <div className="h-[440px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart margin={{ top: 26, right: 26, bottom: 24, left: 5 }}>
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
                          <ZAxis type="number" dataKey="reviews" range={[220, 980]} />
                          <Tooltip
                            cursor={{ strokeDasharray: "3 3" }}
                            content={({ active, payload }) => {
                              if (!active || !payload?.length) return null;
                              const item = payload[0].payload as (typeof companyClusterRows)[number];
                              return (
                                <div className="rounded-lg border border-border bg-card p-3 text-sm shadow-card">
                                  <p className="font-medium text-foreground">{item.label}</p>
                                  <p className="text-muted-foreground">
                                    {item.company} | {item.domain}
                                  </p>
                                  <p className="mt-2 text-muted-foreground">
                                    {item.reviews} reviews | signal {item.intensity}%
                                  </p>
                                  <p className="mt-1 text-muted-foreground">
                                    Terms: {item.terms.join(", ")}
                                  </p>
                                </div>
                              );
                            }}
                          />
                          {companyClusterGroups.map((group) => (
                            <Scatter
                              key={group.company}
                              name={group.company}
                              data={group.rows}
                              fill={group.fill}
                              fillOpacity={0.72}
                            >
                              <LabelList
                                dataKey="label"
                                position="top"
                                offset={8}
                                fill="hsl(var(--foreground))"
                                fontSize={11}
                              />
                              {group.rows.map((entry) => (
                                <Cell key={`${entry.company}-${entry.label}`} fill={entry.fill} />
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
                        Google has a large workload-pressure cluster, while Adobe and
                        Microsoft cluster higher around culture and belonging.
                      </p>
                    </div>
                    <div className="rounded-lg border border-border p-4">
                      <p className="font-medium text-foreground">Cluster terms</p>
                      <div className="mt-3 space-y-3">
                        {companyClusterRows.slice(0, 4).map((cluster) => (
                          <div key={`${cluster.company}-${cluster.label}`}>
                            <p className="text-sm font-medium">{cluster.label}</p>
                            <p className="text-xs text-muted-foreground">
                              {cluster.company}: {cluster.terms.join(", ")}
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
        </section>
      </main>
    </div>
  );
}
