import { useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardCheck,
  Copy,
  Eye,
  FileText,
  Gauge,
  Loader2,
  Save,
  Share2,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

type EvidenceKind = "supporting" | "caution" | "contradictory";
type SandboxMode = "ready" | "loading" | "empty";

const scoreFactors = [
  {
    label: "Need fit",
    value: 34,
    max: 40,
    note: "Daily work evidence aligns with the user's strongest needs: low politics, stable managers, and respectful teams.",
  },
  {
    label: "Theme lift",
    value: 13,
    max: 20,
    note: "Review clusters repeatedly mention manager accessibility, training, and predictable operating norms.",
  },
  {
    label: "Risk drag",
    value: -7,
    max: 15,
    note: "Fast pace and inconsistent location-level staffing reduce the fit for users sensitive to overload.",
  },
  {
    label: "Evidence quality",
    value: 12,
    max: 15,
    note: "The result is supported by multiple recent reviews rather than a single unusually positive quote.",
  },
  {
    label: "Confidence adjustment",
    value: 6,
    max: 10,
    note: "The user profile is specific enough to avoid overfitting to generic positive culture language.",
  },
];

const confidenceFactors = [
  { label: "Profile clarity", value: 84 },
  { label: "Review volume", value: 78 },
  { label: "Evidence agreement", value: 72 },
  { label: "Evidence recency", value: 69 },
  { label: "Data completeness", value: 74 },
];

const evidenceItems: Array<{
  kind: EvidenceKind;
  title: string;
  review: string;
  phrases: string[];
  source: string;
}> = [
  {
    kind: "supporting",
    title: "Why the top match fits",
    review:
      "Managers are present on the floor and usually willing to coach people through problems. The pace is high, but the team culture makes it easier to ask questions and recover from mistakes.",
    phrases: ["willing to coach", "team culture", "ask questions", "recover from mistakes"],
    source: "Current employee, 2024",
  },
  {
    kind: "caution",
    title: "Where the match has tradeoffs",
    review:
      "The work can become intense during peak hours and some locations feel understaffed. People who need a slower rhythm may find the day-to-day environment draining.",
    phrases: ["intense during peak hours", "understaffed", "slower rhythm", "draining"],
    source: "Former employee, 2023",
  },
  {
    kind: "contradictory",
    title: "Evidence that limits certainty",
    review:
      "Corporate values sound consistent, but the actual experience depends on the local leader. One store can feel supportive while another can feel much more political.",
    phrases: ["depends on the local leader", "supportive", "political"],
    source: "Current employee, 2022",
  },
];

const whyNotCompanies = [
  {
    company: "Adobe",
    score: 58,
    reason:
      "Strong on creative autonomy and status, but the evidence shows more ambiguity around role pressure than this user profile seems to tolerate.",
  },
  {
    company: "Apple",
    score: 55,
    reason:
      "High recognition and learning potential, but the review evidence suggests pace and internal competition could conflict with the user's low-politics preference.",
  },
  {
    company: "Optum",
    score: 49,
    reason:
      "Stable processes are a plus, but lower evidence strength around manager trust and day-to-day psychological safety reduces confidence.",
  },
];

const matchCards = [
  {
    company: "Crew Carwash",
    score: 58,
    reviews: "1,284 reviews",
    summary: "Best evidence for low-politics teams, approachable managers, and clear day-to-day expectations.",
  },
  {
    company: "Adobe",
    score: 58,
    reviews: "4,912 reviews",
    summary: "Strong autonomy and recognition, with more ambiguity around pressure and role complexity.",
  },
  {
    company: "Apple",
    score: 55,
    reviews: "8,108 reviews",
    summary: "High growth signal, but pace and internal competition create a sharper tradeoff.",
  },
  {
    company: "Optum",
    score: 49,
    reviews: "6,441 reviews",
    summary: "Operational stability is useful, but manager-trust evidence is less consistent.",
  },
];

const balanceChartData = [
  { label: "Support", value: 68, color: "#315f2b" },
  { label: "Caution", value: 24, color: "#b7791f" },
  { label: "Conflict", value: 18, color: "#9f3a38" },
];

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const highlightReview = (text: string, phrases: string[]) => {
  const pattern = new RegExp(`(${phrases.map(escapeRegExp).join("|")})`, "gi");

  return text.split(pattern).map((part, index) => {
    const isMatch = phrases.some((phrase) => phrase.toLowerCase() === part.toLowerCase());

    if (!isMatch) {
      return <span key={`${part}-${index}`}>{part}</span>;
    }

    return (
      <mark key={`${part}-${index}`} className="rounded bg-[#f3e6b3] px-1 text-[#172819]">
        {part}
      </mark>
    );
  });
};

const kindStyles: Record<EvidenceKind, string> = {
  supporting: "border-[#315f2b]/25 bg-[#eef3e8]",
  caution: "border-[#b7791f]/25 bg-[#fff8e8]",
  contradictory: "border-[#9f3a38]/25 bg-[#fff0ed]",
};

const MatchReportSandbox = () => {
  const [openFactor, setOpenFactor] = useState(scoreFactors[0].label);
  const [mode, setMode] = useState<SandboxMode>("ready");
  const [shareState, setShareState] = useState("Draft report not saved");

  const totalScore = useMemo(() => scoreFactors.reduce((sum, factor) => sum + factor.value, 0), []);
  const confidenceScore = useMemo(
    () => Math.round(confidenceFactors.reduce((sum, factor) => sum + factor.value, 0) / confidenceFactors.length),
    [],
  );

  const simulateLoading = () => {
    setMode("loading");
    window.setTimeout(() => setMode("ready"), 1400);
  };

  const copyShareLink = async () => {
    const url = `${window.location.origin}/match-report-sandbox#demo-report`;
    await navigator.clipboard?.writeText(url);
    setShareState("Demo share link copied");
  };

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="border-b border-border bg-card">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <Button variant="ghost" asChild className="gap-2">
            <Link to="/">
              <ArrowLeft className="h-4 w-4" />
              New Analysis
            </Link>
          </Button>
          <Badge variant="outline" className="border-primary/30 bg-background text-primary">
            Sandbox route: /match-report-sandbox
          </Badge>
        </div>
      </div>

      <section className="mx-auto max-w-7xl px-6 py-10">
        <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="mb-2 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
              Behavioral Match
            </p>
            <h1 className="font-serif text-4xl font-semibold text-foreground md:text-5xl">Your matches</h1>
            <p className="mt-4 max-w-3xl text-base leading-7 text-muted-foreground">
              Production-style sandbox showing how the improved match report would look after a user submits a
              profile. Data is mocked, but the layout mirrors the real results experience.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button variant="outline" onClick={() => setMode("empty")}>
              Empty state
            </Button>
            <Button variant="outline" onClick={simulateLoading}>
              Loading state
            </Button>
            <Button onClick={() => setMode("ready")} className="bg-primary text-primary-foreground hover:bg-primary/90">
              Report state
            </Button>
          </div>
        </div>

        <div className="mb-8 grid gap-4 md:grid-cols-4">
          {matchCards.map((match, index) => (
            <Card key={match.company} className="border-primary/20 transition-all hover:border-primary/40">
              <CardContent className="pt-5">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                      Rank {index + 1}
                    </p>
                    <h2 className="mt-1 font-serif text-xl font-semibold">{match.company}</h2>
                  </div>
                  <Badge className="bg-primary text-primary-foreground">{match.score}%</Badge>
                </div>
                <p className="text-xs text-muted-foreground">{match.reviews}</p>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">{match.summary}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.08fr_0.92fr]">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl">
                <Sparkles className="h-5 w-5 text-primary" />
                Top match: Crew Carwash
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid gap-5 md:grid-cols-[0.34fr_0.66fr]">
                <div className="rounded-2xl border border-primary/20 bg-primary/5 p-5">
                  <p className="text-sm font-medium text-muted-foreground">Overall match</p>
                  <p className="mt-2 font-serif text-6xl font-semibold text-primary">{totalScore}%</p>
                  <p className="mt-3 text-sm leading-6 text-muted-foreground">
                    Strongest current fit, but not an absolute recommendation.
                  </p>
                </div>
                <div>
                  <p className="text-base leading-7 text-muted-foreground">
                    Crew Carwash ranks first because the evidence is strongest around low-politics teamwork,
                    approachable managers, and predictable operating norms. The caution is pace: this fit works better
                    for someone who wants support and standards, not a slow or low-pressure environment.
                  </p>
                  <div className="mt-5 grid gap-3 sm:grid-cols-3">
                    <PlanPill icon={<BarChart3 className="h-4 w-4" />} title="Explain score" />
                    <PlanPill icon={<Eye className="h-4 w-4" />} title="Balanced evidence" />
                    <PlanPill icon={<Share2 className="h-4 w-4" />} title="Share report" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-primary/20 bg-primary/5">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl">
                <Gauge className="h-5 w-5 text-primary" />
                Match confidence meter
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end gap-3">
                <span className="font-serif text-6xl font-semibold text-primary">{confidenceScore}%</span>
                <span className="pb-3 text-sm uppercase tracking-[0.2em] text-muted-foreground">
                  High enough to explain
                </span>
              </div>
              <p className="mt-4 text-sm leading-6 text-muted-foreground">
                Confidence should not mean truth. It should tell the user whether the profile, review volume, and
                evidence agreement are strong enough to trust the direction of the match.
              </p>
              <div className="mt-6 space-y-4">
                {confidenceFactors.map((factor) => (
                  <div key={factor.label}>
                    <div className="mb-2 flex justify-between text-sm text-foreground">
                      <span>{factor.label}</span>
                      <span>{factor.value}%</span>
                    </div>
                    <Progress value={factor.value} className="h-2 bg-primary/10" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {mode === "loading" && <LoadingPreview />}
        {mode === "empty" && <EmptyPreview />}
        {mode === "ready" && (
          <div className="mt-8 grid gap-6">
            <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
              <Card className="border-[#ded8cc] bg-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-2xl">
                    <Sparkles className="h-5 w-5 text-[#315f2b]" />
                    Interactive score explanation
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="rounded-3xl border border-[#ded8cc] bg-[#fbfaf7] p-5">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm uppercase tracking-[0.18em] text-[#756a59]">Crew Carwash match</p>
                        <p className="mt-1 text-5xl font-semibold">{totalScore}%</p>
                      </div>
                      <ShieldCheck className="h-12 w-12 text-[#315f2b]" />
                    </div>
                  </div>
                  <div className="mt-5 space-y-3">
                    {scoreFactors.map((factor) => {
                      const isOpen = openFactor === factor.label;
                      const width = Math.min(100, Math.round((Math.abs(factor.value) / factor.max) * 100));

                      return (
                        <button
                          key={factor.label}
                          type="button"
                          onClick={() => setOpenFactor(isOpen ? "" : factor.label)}
                          className="w-full rounded-2xl border border-[#ded8cc] bg-white p-4 text-left transition hover:border-[#315f2b]/40"
                        >
                          <div className="flex items-center justify-between gap-4">
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center justify-between gap-3">
                                <span className="font-semibold">{factor.label}</span>
                                <span className={factor.value < 0 ? "text-[#9f3a38]" : "text-[#315f2b]"}>
                                  {factor.value > 0 ? "+" : ""}
                                  {factor.value}
                                </span>
                              </div>
                              <div className="mt-3 h-2 rounded-full bg-[#ece7dc]">
                                <div
                                  className={`h-2 rounded-full ${factor.value < 0 ? "bg-[#9f3a38]" : "bg-[#315f2b]"}`}
                                  style={{ width: `${width}%` }}
                                />
                              </div>
                            </div>
                            {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </div>
                          {isOpen && <p className="mt-3 text-sm leading-6 text-[#756a59]">{factor.note}</p>}
                        </button>
                      );
                    })}
                  </div>
                </CardContent>
              </Card>

              <Card className="border-[#ded8cc] bg-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-2xl">
                    <ClipboardCheck className="h-5 w-5 text-[#315f2b]" />
                    Evidence balance and highlighting
                  </CardTitle>
                  <p className="text-sm text-[#756a59]">
                    This becomes the evidence module under clusters in single-company analysis and inside match reports.
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 lg:grid-cols-[0.75fr_1.25fr]">
                    <div className="rounded-3xl border border-[#ded8cc] bg-[#fbfaf7] p-4">
                      <p className="mb-4 text-sm font-semibold uppercase tracking-[0.18em] text-[#756a59]">
                        Evidence mix
                      </p>
                      <ResponsiveContainer width="100%" height={220}>
                        <BarChart data={balanceChartData}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="label" tickLine={false} axisLine={false} />
                          <YAxis hide domain={[0, 80]} />
                          <Tooltip />
                          <Bar dataKey="value" radius={[10, 10, 0, 0]}>
                            {balanceChartData.map((entry) => (
                              <Cell key={entry.label} fill={entry.color} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="space-y-3">
                      {evidenceItems.map((item) => (
                        <article key={item.title} className={`rounded-2xl border p-4 ${kindStyles[item.kind]}`}>
                          <div className="flex items-center justify-between gap-3">
                            <h3 className="font-semibold">{item.title}</h3>
                            <Badge variant="outline" className="capitalize">
                              {item.kind}
                            </Badge>
                          </div>
                          <p className="mt-3 text-sm leading-6 text-[#5f574b]">{highlightReview(item.review, item.phrases)}</p>
                          <p className="mt-3 text-xs uppercase tracking-[0.16em] text-[#756a59]">{item.source}</p>
                        </article>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </section>

            <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
              <Card className="border-[#ded8cc] bg-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-2xl">
                    <XCircle className="h-5 w-5 text-[#9f3a38]" />
                    Why not these companies?
                  </CardTitle>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-3">
                  {whyNotCompanies.map((company) => (
                    <div key={company.company} className="rounded-3xl border border-[#ded8cc] bg-[#fbfaf7] p-5">
                      <div className="flex items-start justify-between gap-3">
                        <h3 className="text-xl font-semibold">{company.company}</h3>
                        <Badge variant="outline">{company.score}%</Badge>
                      </div>
                      <p className="mt-4 text-sm leading-6 text-[#756a59]">{company.reason}</p>
                    </div>
                  ))}
                </CardContent>
              </Card>

              <Card className="border-[#ded8cc] bg-white">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-2xl">
                    <Save className="h-5 w-5 text-[#315f2b]" />
                    Save and share match report
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="rounded-3xl border border-[#ded8cc] bg-[#fbfaf7] p-5">
                    <p className="text-sm uppercase tracking-[0.18em] text-[#756a59]">Report contents</p>
                    <ul className="mt-4 space-y-3 text-sm text-[#5f574b]">
                      <li className="flex gap-2">
                        <CheckCircle2 className="mt-0.5 h-4 w-4 text-[#315f2b]" />
                        User profile summary and match confidence
                      </li>
                      <li className="flex gap-2">
                        <CheckCircle2 className="mt-0.5 h-4 w-4 text-[#315f2b]" />
                        Top company ranking with balanced evidence
                      </li>
                      <li className="flex gap-2">
                        <CheckCircle2 className="mt-0.5 h-4 w-4 text-[#315f2b]" />
                        Why-not explanations for companies that lost
                      </li>
                    </ul>
                  </div>
                  <div className="mt-5 flex flex-wrap gap-3">
                    <Button className="gap-2 bg-[#315f2b] text-white hover:bg-[#264c22]" onClick={copyShareLink}>
                      <Copy className="h-4 w-4" />
                      Copy share link
                    </Button>
                    <Button variant="outline" className="gap-2" onClick={() => setShareState("Demo report saved locally")}>
                      <FileText className="h-4 w-4" />
                      Save report
                    </Button>
                  </div>
                  <p className="mt-4 text-sm text-[#756a59]">{shareState}</p>
                </CardContent>
              </Card>
            </section>
          </div>
        )}
      </section>
    </main>
  );
};

const PlanPill = ({ icon, title }: { icon: React.ReactNode; title: string }) => (
  <div className="flex items-center gap-3 rounded-2xl border border-[#ded8cc] bg-[#fbfaf7] p-4 text-sm font-semibold">
    <span className="rounded-full bg-[#e8eadf] p-2 text-[#315f2b]">{icon}</span>
    {title}
  </div>
);

const LoadingPreview = () => (
  <Card className="mt-8 border-[#ded8cc] bg-white">
    <CardContent className="flex flex-col items-center justify-center px-6 py-16 text-center">
      <Loader2 className="h-10 w-10 animate-spin text-[#315f2b]" />
      <h2 className="mt-5 text-2xl font-semibold">Building your evidence-backed match report</h2>
      <p className="mt-3 max-w-2xl text-[#756a59]">
        The production flow should name the work being done: validating profile quality, scoring companies, balancing
        review evidence, and generating plain-language explanations.
      </p>
      <div className="mt-6 grid w-full max-w-3xl gap-3 md:grid-cols-4">
        {["Profile quality", "Company scoring", "Evidence balance", "Report writing"].map((step) => (
          <div key={step} className="rounded-2xl border border-[#ded8cc] bg-[#fbfaf7] p-4 text-sm">
            {step}
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
);

const EmptyPreview = () => (
  <Card className="mt-8 border-[#ded8cc] bg-white">
    <CardContent className="grid gap-5 px-6 py-12 md:grid-cols-3">
      <div className="rounded-3xl border border-[#ded8cc] bg-[#fbfaf7] p-6">
        <AlertTriangle className="h-8 w-8 text-[#b7791f]" />
        <h2 className="mt-4 text-xl font-semibold">Not enough profile signal</h2>
        <p className="mt-3 text-sm leading-6 text-[#756a59]">
          Ask for a concrete workplace situation, not just goals. Example: "I want low politics and clear managers"
          is weaker than a narrative about what went wrong before.
        </p>
      </div>
      <div className="rounded-3xl border border-[#ded8cc] bg-[#fbfaf7] p-6">
        <Eye className="h-8 w-8 text-[#315f2b]" />
        <h2 className="mt-4 text-xl font-semibold">No balanced evidence found</h2>
        <p className="mt-3 text-sm leading-6 text-[#756a59]">
          If the company has only positive or only negative snippets, show that limitation instead of pretending the
          report is balanced.
        </p>
      </div>
      <div className="rounded-3xl border border-[#ded8cc] bg-[#fbfaf7] p-6">
        <Share2 className="h-8 w-8 text-[#315f2b]" />
        <h2 className="mt-4 text-xl font-semibold">Report not shareable yet</h2>
        <p className="mt-3 text-sm leading-6 text-[#756a59]">
          A real share link needs persisted report inputs, model version, evidence IDs, and generated summary text.
        </p>
      </div>
    </CardContent>
  </Card>
);

export default MatchReportSandbox;
