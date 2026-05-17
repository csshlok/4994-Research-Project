import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  BarChart3,
  Brain,
  Check,
  FileText,
  LineChart,
  Search,
  ShieldAlert,
  Sparkles,
  Target,
  TrendingUp,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";

type DomainKey = "phys" | "selfprot" | "aff" | "stat" | "fam";
type ToggleKey =
  | "avoid_burnout"
  | "avoid_toxicity"
  | "growth"
  | "flexibility"
  | "belonging"
  | "stability"
  | "compensation";

interface Domain {
  key: DomainKey;
  label: string;
  short: string;
  description: string;
}

interface CompanyProfile {
  id: string;
  label: string;
  reviews: number;
  domains: Record<DomainKey, number>;
  fulfillment: Array<{ label: string; domain: DomainKey; signal: number; terms: string[] }>;
  risks: Array<{ label: string; domain: DomainKey; signal: number; terms: string[] }>;
  ragSummary: string;
  ragStrengths: string[];
  ragRisks: string[];
}

interface InterpretedProfile {
  weights: Record<DomainKey, number>;
  desiredThemes: string[];
  avoidThemes: string[];
  summary: string;
}

const DOMAINS: Domain[] = [
  {
    key: "phys",
    label: "Physiological",
    short: "Pay",
    description: "pay, benefits, workload, pace, and basic support",
  },
  {
    key: "selfprot",
    label: "Self-Protection",
    short: "Safety",
    description: "fairness, trust, stability, respect, and psychological safety",
  },
  {
    key: "aff",
    label: "Affiliation",
    short: "Belonging",
    description: "team connection, collaboration, culture, and inclusion",
  },
  {
    key: "stat",
    label: "Status & Esteem",
    short: "Growth",
    description: "recognition, advancement, feedback, and career momentum",
  },
  {
    key: "fam",
    label: "Family Care",
    short: "Flex",
    description: "flexibility, work-life balance, scheduling, and care support",
  },
];

const TOGGLES: Array<{ key: ToggleKey; label: string; weights: Partial<Record<DomainKey, number>>; avoid?: string[]; desired?: string[] }> = [
  {
    key: "avoid_burnout",
    label: "Avoid burnout",
    weights: { phys: 2, fam: 2 },
    avoid: ["burnout", "long hours", "unsustainable workload"],
  },
  {
    key: "avoid_toxicity",
    label: "Avoid toxic management",
    weights: { selfprot: 3, aff: 1 },
    avoid: ["toxic leadership", "retaliation", "favoritism"],
  },
  {
    key: "growth",
    label: "Prioritize growth",
    weights: { stat: 3 },
    desired: ["career growth", "mentorship", "recognition"],
  },
  {
    key: "flexibility",
    label: "Prioritize flexibility",
    weights: { fam: 3 },
    desired: ["remote work", "flexible schedule", "work-life balance"],
  },
  {
    key: "belonging",
    label: "Prioritize belonging",
    weights: { aff: 3, selfprot: 1 },
    desired: ["supportive coworkers", "collaboration", "inclusive culture"],
  },
  {
    key: "stability",
    label: "Prioritize stability",
    weights: { selfprot: 2, phys: 1 },
    desired: ["job security", "clear leadership", "reliable organization"],
  },
  {
    key: "compensation",
    label: "Prioritize compensation",
    weights: { phys: 3, stat: 1 },
    desired: ["strong pay", "benefits", "bonus"],
  },
];

const COMPANIES: CompanyProfile[] = [
  {
    id: "adobe",
    label: "Adobe",
    reviews: 620,
    domains: { phys: 78, selfprot: 77, aff: 86, stat: 80, fam: 79 },
    fulfillment: [
      { label: "Respectful Culture", domain: "aff", signal: 86, terms: ["supportive", "inclusive", "friendly"] },
      { label: "Balanced Flexibility", domain: "fam", signal: 79, terms: ["flexibility", "pto", "balance"] },
      { label: "Growth & Recognition", domain: "stat", signal: 80, terms: ["growth", "recognition", "learning"] },
    ],
    risks: [
      { label: "Large-company process", domain: "stat", signal: 24, terms: ["process", "slow decisions", "bureaucracy"] },
    ],
    ragSummary:
      "Adobe is a strong behavioral fit for users who value supportive culture, balanced work rhythms, and steady recognition. Cached summaries would likely position it as a lower-friction option for people seeking psychological safety and belonging.",
    ragStrengths: ["Employees describe strong belonging and respect signals.", "Flexibility and balance show up as recurring positive themes."],
    ragRisks: ["Some reviews point to process overhead and slower decision making."],
  },
  {
    id: "microsoft",
    label: "Microsoft",
    reviews: 777,
    domains: { phys: 61, selfprot: 50, aff: 55, stat: 58, fam: 60 },
    fulfillment: [
      { label: "Pay & Benefits", domain: "phys", signal: 33, terms: ["good pay", "benefits", "bonus"] },
      { label: "Flexibility & Care", domain: "fam", signal: 27, terms: ["work life balance", "flexible", "pto"] },
      { label: "Growth & Recognition", domain: "stat", signal: 22, terms: ["promotion", "recognition", "advancement"] },
    ],
    risks: [
      { label: "Bureaucracy & Change", domain: "selfprot", signal: 35, terms: ["bureaucracy", "management", "reorg"] },
      { label: "Workload Pressure", domain: "phys", signal: 28, terms: ["stress", "long hours", "burnout"] },
    ],
    ragSummary:
      "Microsoft can be a good fit for users who want brand scale, benefits, and career pathways, but the match depends on tolerance for organizational complexity. Cached RAG evidence would call out both strong resources and notable friction around workload or management variance.",
    ragStrengths: ["Benefits, growth, and flexibility appear in positive clusters.", "The company offers broad career surface area."],
    ragRisks: ["Users sensitive to bureaucracy, manager inconsistency, or workload pressure should inspect the risks closely."],
  },
  {
    id: "google",
    label: "Google",
    reviews: 980,
    domains: { phys: 84, selfprot: 66, aff: 76, stat: 75, fam: 61 },
    fulfillment: [
      { label: "Compensation & Perks", domain: "phys", signal: 84, terms: ["salary", "benefits", "perks"] },
      { label: "Smart Collaborative Teams", domain: "aff", signal: 76, terms: ["smart people", "teamwork", "collaboration"] },
      { label: "Learning Environment", domain: "stat", signal: 75, terms: ["learning", "visibility", "projects"] },
    ],
    risks: [
      { label: "Work-Life Pressure", domain: "fam", signal: 42, terms: ["long hours", "always on", "pressure"] },
      { label: "Competitive Environment", domain: "selfprot", signal: 30, terms: ["politics", "competition", "performance"] },
    ],
    ragSummary:
      "Google is likely to match users who prioritize compensation, ambitious peers, and high-resource environments. It may be a weaker behavioral fit for someone who puts sustainable pace and predictable boundaries above prestige or compensation.",
    ragStrengths: ["Compensation and learning signals are comparatively strong.", "Collaboration and high-talent peer language appears frequently."],
    ragRisks: ["Work-life pressure is the most important caution for balance-sensitive users."],
  },
  {
    id: "sap",
    label: "SAP",
    reviews: 720,
    domains: { phys: 76, selfprot: 73, aff: 78, stat: 63, fam: 82 },
    fulfillment: [
      { label: "Work-Life Balance", domain: "fam", signal: 82, terms: ["balance", "remote", "flexible hours"] },
      { label: "Supportive Teams", domain: "aff", signal: 78, terms: ["colleagues", "supportive", "helpful"] },
      { label: "Benefits", domain: "phys", signal: 76, terms: ["benefits", "leave", "insurance"] },
    ],
    risks: [
      { label: "Slow Advancement", domain: "stat", signal: 37, terms: ["slow growth", "promotion", "legacy"] },
    ],
    ragSummary:
      "SAP is a strong fit for users who value flexibility, stable teams, and sustainable work over fast advancement. Cached evidence would likely frame it as a balanced option with some risk of slower career acceleration.",
    ragStrengths: ["Flexibility and work-life balance are core positive signals.", "Team support appears more consistently than competitive pressure."],
    ragRisks: ["Ambitious users may find advancement or innovation pace too slow."],
  },
  {
    id: "deloitte",
    label: "Deloitte",
    reviews: 850,
    domains: { phys: 68, selfprot: 59, aff: 69, stat: 82, fam: 51 },
    fulfillment: [
      { label: "Career Acceleration", domain: "stat", signal: 82, terms: ["promotion", "learning", "client exposure"] },
      { label: "Professional Network", domain: "aff", signal: 69, terms: ["network", "team", "mentors"] },
    ],
    risks: [
      { label: "Long Hours", domain: "fam", signal: 52, terms: ["long hours", "travel", "weekends"] },
      { label: "Pressure & Politics", domain: "selfprot", signal: 38, terms: ["pressure", "politics", "utilization"] },
    ],
    ragSummary:
      "Deloitte is a stronger behavioral match for people who explicitly want growth, client exposure, and fast career signaling. It is a weaker fit for users whose language emphasizes boundaries, stability, and low-burnout environments.",
    ragStrengths: ["Career growth and professional learning signals are strong.", "Network-building language is a positive theme."],
    ragRisks: ["Workload, travel, and pressure are meaningful risks for balance-sensitive users."],
  },
];

const SAMPLE_GOAL =
  "I want a company where managers are fair, people are collaborative, and I can grow without burning out. I care less about prestige and more about stability, flexibility, and feeling respected.";

const MATCH_PROCESS = [
  {
    step: "01",
    title: "Interpret",
    detail: "Convert narrative and controls into a behavioral need profile.",
  },
  {
    step: "02",
    title: "Weight",
    detail: "Balance five workplace goals and separate needs from risks.",
  },
  {
    step: "03",
    title: "Rank",
    detail: "Compare the profile against company-level review evidence.",
  },
  {
    step: "04",
    title: "Explain",
    detail: "Surface the matched evidence, penalties, and fit caveats.",
  },
];

function emptyWeights(): Record<DomainKey, number> {
  return { phys: 1, selfprot: 1, aff: 1, stat: 1, fam: 1 };
}

function normalizeWeights(raw: Record<DomainKey, number>): Record<DomainKey, number> {
  const total = Object.values(raw).reduce((sum, value) => sum + value, 0) || 1;
  return Object.fromEntries(
    DOMAINS.map((domain) => [domain.key, raw[domain.key] / total])
  ) as Record<DomainKey, number>;
}

function includesAny(text: string, words: string[]) {
  return words.some((word) => text.includes(word));
}

function interpretGoals(goalText: string, selectedToggles: ToggleKey[]): InterpretedProfile {
  const text = goalText.toLowerCase();
  const weights = emptyWeights();
  const desiredThemes = new Set<string>();
  const avoidThemes = new Set<string>();

  if (includesAny(text, ["pay", "salary", "benefit", "compensation", "bonus", "healthcare"])) {
    weights.phys += 3;
    desiredThemes.add("strong pay and benefits");
  }
  if (includesAny(text, ["fair", "safe", "respect", "stable", "security", "trust", "manager"])) {
    weights.selfprot += 3;
    desiredThemes.add("fair and respectful management");
  }
  if (includesAny(text, ["team", "people", "collaborative", "belong", "culture", "inclusive"])) {
    weights.aff += 3;
    desiredThemes.add("collaborative culture");
  }
  if (includesAny(text, ["grow", "career", "promotion", "learn", "recognition", "mentor"])) {
    weights.stat += 3;
    desiredThemes.add("career growth and recognition");
  }
  if (includesAny(text, ["flex", "remote", "balance", "family", "schedule", "burnout", "hours"])) {
    weights.fam += 3;
    desiredThemes.add("flexibility and sustainable pace");
  }
  if (includesAny(text, ["burnout", "overwork", "long hours", "stress", "weekend"])) {
    avoidThemes.add("burnout and long hours");
  }
  if (includesAny(text, ["toxic", "politics", "favoritism", "retaliation", "unfair"])) {
    avoidThemes.add("toxic or unfair management");
  }

  selectedToggles.forEach((toggleKey) => {
    const toggle = TOGGLES.find((item) => item.key === toggleKey);
    if (!toggle) return;
    Object.entries(toggle.weights).forEach(([key, value]) => {
      weights[key as DomainKey] += value || 0;
    });
    toggle.desired?.forEach((item) => desiredThemes.add(item));
    toggle.avoid?.forEach((item) => avoidThemes.add(item));
  });

  return {
    weights: normalizeWeights(weights),
    desiredThemes: Array.from(desiredThemes).slice(0, 8),
    avoidThemes: Array.from(avoidThemes).slice(0, 8),
    summary:
      "The user appears to be describing workplace fit through needs, boundaries, and motivational context rather than job-function matching.",
  };
}

function themeHit(profile: InterpretedProfile, company: CompanyProfile) {
  const desired = profile.desiredThemes.join(" ").toLowerCase();
  const avoid = profile.avoidThemes.join(" ").toLowerCase();
  const positive = company.fulfillment.reduce((sum, cluster) => {
    const clusterText = `${cluster.label} ${cluster.terms.join(" ")}`.toLowerCase();
    return sum + (desired.split(" ").some((word) => word.length > 4 && clusterText.includes(word)) ? cluster.signal : 0);
  }, 0);
  const risk = company.risks.reduce((sum, cluster) => {
    const clusterText = `${cluster.label} ${cluster.terms.join(" ")}`.toLowerCase();
    return sum + (avoid.split(" ").some((word) => word.length > 4 && clusterText.includes(word)) ? cluster.signal : 0);
  }, 0);
  return { positive, risk };
}

function scoreMatch(profile: InterpretedProfile, company: CompanyProfile) {
  const weightedDomain = DOMAINS.reduce(
    (sum, domain) => sum + company.domains[domain.key] * profile.weights[domain.key],
    0
  );
  const hits = themeHit(profile, company);
  const themeBonus = Math.min(10, hits.positive / 30);
  const riskPenalty = Math.min(18, hits.risk / 8);
  const confidence = Math.min(5, Math.log10(company.reviews) * 1.5);
  const score = Math.round(Math.max(0, Math.min(100, weightedDomain + themeBonus + confidence - riskPenalty)));
  return { score, weightedDomain: Math.round(weightedDomain), themeBonus: Math.round(themeBonus), riskPenalty: Math.round(riskPenalty) };
}

function scoreBadge(score: number) {
  if (score >= 80) return "bg-primary text-primary-foreground";
  if (score >= 70) return "bg-olive-muted/80 text-foreground";
  return "bg-destructive/15 text-destructive";
}

export default function MatchSandbox() {
  const [goalText, setGoalText] = useState(SAMPLE_GOAL);
  const [selectedToggles, setSelectedToggles] = useState<ToggleKey[]>(["avoid_burnout", "avoid_toxicity", "flexibility"]);
  const [selectedCompanyId, setSelectedCompanyId] = useState("adobe");

  const profile = useMemo(() => interpretGoals(goalText, selectedToggles), [goalText, selectedToggles]);
  const matches = useMemo(
    () =>
      COMPANIES.map((company) => ({ company, ...scoreMatch(profile, company) })).sort(
        (a, b) => b.score - a.score
      ),
    [profile]
  );
  const selected = matches.find((match) => match.company.id === selectedCompanyId) || matches[0];

  const radarRows = DOMAINS.map((domain) => ({
    domain: domain.short,
    userNeed: Math.round(profile.weights[domain.key] * 100),
    company: selected.company.domains[domain.key],
  }));
  const scoreRows = [
    { label: "Weighted domains", value: selected.weightedDomain, fill: "hsl(var(--primary))" },
    { label: "Theme bonus", value: selected.themeBonus, fill: "hsl(var(--olive))" },
    { label: "Risk penalty", value: -selected.riskPenalty, fill: "hsl(var(--destructive))" },
  ];

  const toggleSelection = (key: ToggleKey) => {
    setSelectedToggles((cur) =>
      cur.includes(key) ? cur.filter((item) => item !== key) : [...cur, key]
    );
  };

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
          <Badge variant="outline">Find My Match Sandbox</Badge>
        </div>
      </header>

      <main className="container-wide py-10 md:py-14">
        <section className="mb-10 max-w-4xl">
          <p className="mb-3 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
            Behavioral Company Match
          </p>
          <h1 className="font-serif text-4xl font-semibold text-foreground md:text-5xl">
            Your matches
          </h1>
          <p className="mt-4 text-lg leading-relaxed text-muted-foreground">
            A local sandbox for shaping the employee-to-company match flow before
            connecting it to the production company artifacts.
          </p>
        </section>

        <section className="mb-8 grid gap-3 md:grid-cols-4">
          {MATCH_PROCESS.map((item) => (
            <div key={item.step} className="rounded-lg border border-border bg-card/70 p-4 shadow-card">
              <div className="mb-3 flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-[0.16em] text-primary">
                  {item.step}
                </span>
                <span className="h-2 w-2 rounded-full bg-primary" />
              </div>
              <h2 className="mb-2 font-serif text-xl font-semibold text-foreground">{item.title}</h2>
              <p className="text-sm leading-relaxed text-muted-foreground">{item.detail}</p>
            </div>
          ))}
        </section>

        <section className="mb-8 grid gap-6 lg:grid-cols-[0.92fr_1.08fr]">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Search className="h-5 w-5" />
                What do you want from a company?
              </CardTitle>
            </CardHeader>
            <CardContent>
              <Textarea
                value={goalText}
                onChange={(event) => setGoalText(event.target.value)}
                className="min-h-[170px] resize-none bg-background text-base leading-relaxed"
                placeholder="Describe the culture, support, pace, and growth environment you want."
              />
              <div className="mt-5 flex flex-wrap gap-2">
                {TOGGLES.map((toggle) => {
                  const active = selectedToggles.includes(toggle.key);
                  return (
                    <button
                      key={toggle.key}
                      type="button"
                      onClick={() => toggleSelection(toggle.key)}
                      className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm transition-colors ${
                        active
                          ? "border-primary bg-primary text-primary-foreground"
                          : "border-border bg-card hover:border-primary/40"
                      }`}
                    >
                      {active ? <Check className="h-4 w-4" /> : <Target className="h-4 w-4" />}
                      {toggle.label}
                    </button>
                  );
                })}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Interpreted Behavioral Profile
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-5 text-sm leading-relaxed text-muted-foreground">
                {profile.summary} In production, Gemini would convert the free text into this
                structured profile, with a local keyword fallback for reliability.
              </p>
              <div className="mb-5 grid gap-3">
                {DOMAINS.map((domain) => (
                  <div key={domain.key}>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span className="font-medium">{domain.label}</span>
                      <span className="text-muted-foreground">{Math.round(profile.weights[domain.key] * 100)}%</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${Math.round(profile.weights[domain.key] * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-border p-4">
                  <p className="mb-3 text-sm font-medium text-primary">Desired themes</p>
                  <div className="flex flex-wrap gap-2">
                    {profile.desiredThemes.map((theme) => (
                      <Badge key={theme} variant="secondary">{theme}</Badge>
                    ))}
                  </div>
                </div>
                <div className="rounded-lg border border-border p-4">
                  <p className="mb-3 text-sm font-medium text-destructive">Sensitive risks</p>
                  <div className="flex flex-wrap gap-2">
                    {profile.avoidThemes.map((theme) => (
                      <Badge key={theme} variant="outline">{theme}</Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="mb-8 grid gap-4 md:grid-cols-5">
          {matches.map((match) => (
            <Card
              key={match.company.id}
              onClick={() => setSelectedCompanyId(match.company.id)}
              className={`cursor-pointer transition-all ${
                selected.company.id === match.company.id ? "border-primary shadow-elevated" : "hover:border-primary/40"
              }`}
            >
              <CardContent className="pt-5">
                <div className="mb-3 flex items-center justify-between gap-2">
                  <h3 className="font-medium">{match.company.label}</h3>
                  <Badge className={scoreBadge(match.score)}>{match.score}%</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  {match.company.reviews.toLocaleString()} cached reviews
                </p>
              </CardContent>
            </Card>
          ))}
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  {selected.company.label} Fit Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <p className="text-muted-foreground leading-relaxed">{selected.company.ragSummary}</p>
                <div className="grid gap-3">
                  <div className="rounded-lg bg-primary/10 p-4">
                    <p className="mb-2 flex items-center gap-2 font-medium text-primary">
                      <TrendingUp className="h-4 w-4" />
                      Why it matches
                    </p>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      {selected.company.ragStrengths.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="rounded-lg bg-destructive/10 p-4">
                    <p className="mb-2 flex items-center gap-2 font-medium text-destructive">
                      <ShieldAlert className="h-4 w-4" />
                      Where it may disappoint
                    </p>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      {selected.company.ragRisks.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Evidence Presented
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  <div>
                    <p className="mb-3 text-sm font-medium">Fulfillment clusters used</p>
                    <div className="grid gap-3">
                      {selected.company.fulfillment.map((cluster) => (
                        <div key={cluster.label} className="rounded-lg border border-border p-3">
                          <div className="mb-2 flex items-center justify-between">
                            <span className="font-medium">{cluster.label}</span>
                            <Badge variant="secondary">{cluster.signal}% signal</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">{cluster.terms.join(", ")}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div>
                    <p className="mb-3 text-sm font-medium">Risk clusters used</p>
                    <div className="grid gap-3">
                      {selected.company.risks.map((cluster) => (
                        <div key={cluster.label} className="rounded-lg border border-border p-3">
                          <div className="mb-2 flex items-center justify-between">
                            <span className="font-medium">{cluster.label}</span>
                            <Badge variant="outline">{cluster.signal}% risk</Badge>
                          </div>
                          <p className="text-sm text-muted-foreground">{cluster.terms.join(", ")}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <LineChart className="h-5 w-5" />
                  Need Alignment
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={radarRows} outerRadius="76%">
                      <PolarGrid stroke="hsl(var(--border))" />
                      <PolarAngleAxis dataKey="domain" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Radar
                        name="User need weight"
                        dataKey="userNeed"
                        stroke="hsl(var(--olive))"
                        fill="hsl(var(--olive))"
                        fillOpacity={0.18}
                        strokeWidth={2}
                      />
                      <Radar
                        name={selected.company.label}
                        dataKey="company"
                        stroke="hsl(var(--primary))"
                        fill="hsl(var(--primary))"
                        fillOpacity={0.25}
                        strokeWidth={2}
                      />
                      <Tooltip />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Match Score Breakdown
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={scoreRows} margin={{ top: 8, right: 16, bottom: 18, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis dataKey="label" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <YAxis domain={[-20, 100]} tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }} />
                      <Tooltip />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {scoreRows.map((row) => (
                          <Cell key={row.label} fill={row.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 rounded-lg border border-border bg-muted/40 p-3 text-sm text-muted-foreground">
                  Production scoring would use cached `company_scores.csv`, `topic_summary.csv`,
                  `rag_summary.json`, `rag_clusters.json`, and `rag_insights.json`. Gemini would
                  explain the top matches after deterministic ranking is complete.
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Domain Score Table</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[520px] text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-muted-foreground">
                        <th className="py-2 font-medium">Domain</th>
                        <th className="py-2 font-medium">User Weight</th>
                        <th className="py-2 font-medium">{selected.company.label}</th>
                        <th className="py-2 font-medium">Meaning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {DOMAINS.map((domain) => (
                        <tr key={domain.key} className="border-b border-border/70">
                          <td className="py-3 font-medium">{domain.label}</td>
                          <td className="py-3">{Math.round(profile.weights[domain.key] * 100)}%</td>
                          <td className="py-3">{selected.company.domains[domain.key]}%</td>
                          <td className="py-3 text-muted-foreground">{domain.description}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>
      </main>
    </div>
  );
}
