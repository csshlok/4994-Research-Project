import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  MATCH_DOMAINS,
  scoreClass,
  type CompanyMatchResult,
  type MatchDomainKey,
  type MatchEvidence,
  type MatchProfile,
} from "@/lib/matching";
import { type MatchTopSummary } from "@/lib/backend-api";
import {
  BarChart3,
  ChevronDown,
  ChevronUp,
  FileText,
  Gauge,
  ShieldAlert,
  Sparkles,
  TrendingUp,
  XCircle,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

interface MatchResultsViewProps {
  title: string;
  profile: MatchProfile;
  results: CompanyMatchResult[];
  limit?: number;
  onOpenCompany?: (companyId: string) => void;
  topSummary?: MatchTopSummary | null;
}

export function MatchResultsView({
  title,
  profile,
  results,
  limit = 4,
  onOpenCompany,
  topSummary,
}: MatchResultsViewProps) {
  const visibleResults = results.slice(0, limit);
  const isComparison = visibleResults.length > 1;
  const topResult = visibleResults[0];
  const firstVisibleId = visibleResults[0]?.company.id || "";
  const [selectedId, setSelectedId] = useState<string>("");
  const [openScoreFactor, setOpenScoreFactor] = useState<string>("Need fit");
  const selectedResult =
    selectedId ? visibleResults.find((result) => result.company.id === selectedId) : undefined;
  const detailResult = selectedResult || visibleResults[0];

  useEffect(() => {
    setSelectedId("");
  }, [firstVisibleId]);

  const tradeoffRows = useMemo(
    () =>
      MATCH_DOMAINS.map((domain) => {
        const ranked = [...visibleResults].sort(
          (a, b) => (b.company.domains[domain.key] || 0) - (a.company.domains[domain.key] || 0)
        );
        return {
          domain,
          priority: Math.round(profile.weights[domain.key] * 100),
          leader: ranked[0],
          trailer: ranked[ranked.length - 1],
          top: topResult,
          current: detailResult,
        };
      }).sort((a, b) => b.priority - a.priority),
    [profile.weights, detailResult, topResult, visibleResults]
  );

  const scoreFactors = useMemo(() => buildScoreFactors(detailResult), [detailResult]);
  const confidenceFactors = useMemo(
    () => buildConfidenceFactors(profile, detailResult),
    [profile, detailResult]
  );
  const confidenceScore = useMemo(
    () =>
      Math.round(
        confidenceFactors.reduce((sum, factor) => sum + factor.value, 0) /
          Math.max(1, confidenceFactors.length)
      ),
    [confidenceFactors]
  );
  const highlightTerms = useMemo(
    () => buildHighlightTerms(profile, detailResult),
    [profile, detailResult]
  );
  const whyNotCompanies = useMemo(
    () => visibleResults.filter((result) => result.company.id !== detailResult?.company.id).slice(0, 3),
    [detailResult?.company.id, visibleResults]
  );

  if (!detailResult) {
    return null;
  }

  return (
    <div className="grid gap-6">
      <section>
        <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="mb-2 text-sm font-semibold uppercase tracking-[0.18em] text-primary">
              Behavioral Match
            </p>
            <h1 className="font-serif text-4xl font-semibold text-foreground md:text-5xl">
              {title}
            </h1>
          </div>
        </div>

        <div className="grid gap-5 md:grid-cols-4">
          {visibleResults.map((result, index) => (
            <Card
              key={result.company.id}
              onClick={() => setSelectedId(result.company.id)}
              className={`min-h-[220px] cursor-pointer border-primary/20 transition-all ${
                selectedResult?.company.id === result.company.id
                  ? "border-primary shadow-elevated"
                  : "hover:border-primary/40"
              }`}
            >
              <CardContent className="p-6">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.28em] text-muted-foreground">
                      Rank {index + 1}
                    </p>
                    <h2 className="mt-2 font-serif text-2xl font-semibold leading-tight">
                      {result.company.label}
                    </h2>
                  </div>
                  <Badge className={scoreClass(result.score)}>{result.score}%</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  {result.company.reviews.toLocaleString()} reviews
                </p>
                <p className="mt-4 text-sm leading-6 text-muted-foreground">
                  {result.summary}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {isComparison ? (
        <section className="grid gap-6 lg:grid-cols-[1fr_0.9fr]">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Match overview
              </CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4">
              {visibleResults.map((result, index) => (
                <button
                  key={result.company.id}
                  type="button"
                  onClick={() => setSelectedId(result.company.id)}
                  className="rounded-xl border border-border bg-background p-4 text-left transition-colors hover:border-primary/50"
                >
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
                        Rank {index + 1}
                      </p>
                      <h2 className="font-serif text-2xl font-semibold">{result.company.label}</h2>
                    </div>
                    <Badge className={scoreClass(result.score)}>{result.score}%</Badge>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-4">
                    {MATCH_DOMAINS.map((domain) => (
                      <div key={domain.key}>
                        <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
                          <span>{domain.short}</span>
                          <span>{result.company.domains[domain.key]}</span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${result.company.domains[domain.key] || 0}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </button>
              ))}
            </CardContent>
          </Card>

          {topResult ? (
            <Card className="border-primary/25 bg-primary/5">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="h-5 w-5" />
                  Top match: {topResult.company.label}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="mb-4 text-5xl font-serif font-semibold text-primary">
                  {topResult.score}%
                </div>
                <p className="font-medium">{topSummary?.headline || `${topResult.company.label} is your strongest current match`}</p>
                <p className="mt-3 text-sm leading-6 text-muted-foreground">
                  {topSummary?.summary || topResult.summary}
                </p>
                <div className="mt-5 grid gap-4">
                  <div>
                    <p className="mb-2 text-sm font-medium">Why it ranked first</p>
                    <div className="grid gap-2 text-sm text-muted-foreground">
                      {(topSummary?.match_reasons?.length ? topSummary.match_reasons : topResult.whyFit).map((item) => (
                        <p key={item}>{item}</p>
                      ))}
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {topSummary?.caveat || "This is a review-derived behavioral fit signal. Select any company above to inspect detailed evidence."}
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : null}
        </section>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Your Behaviour Profile</CardTitle>
        </CardHeader>
        <CardContent className="p-6 md:p-8">
          <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_minmax(420px,0.82fr)]">
            <div className="max-w-3xl space-y-5">
              <p className="text-base leading-8 text-muted-foreground">
                {profile.behaviorProfile.summary}
              </p>
              <p className="text-base leading-8 text-muted-foreground">
                {profile.behaviorProfile.workStyle}
              </p>
            </div>
            <div className="grid gap-4">
              <div className="rounded-xl border border-border p-5">
                <p className="mb-2 text-sm font-medium">Core needs</p>
                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
                  {profile.behaviorProfile.coreNeeds.map((need) => (
                    <Badge key={need} variant="secondary" className="justify-start whitespace-normal px-3 py-1.5 text-left">
                      {need}
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="rounded-xl border border-border p-5">
                <p className="mb-2 text-sm font-medium">Risk sensitivities</p>
                <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
                  {profile.behaviorProfile.riskSensitivities.map((risk) => (
                    <Badge key={risk} variant="outline" className="justify-start whitespace-normal px-3 py-1.5 text-left">
                      {risk}
                    </Badge>
                  ))}
                </div>
              </div>
              <div className="rounded-xl border border-border p-5">
                <p className="mb-2 text-sm font-medium">Likely motivators</p>
                <div className="flex flex-wrap gap-2">
                  {profile.behaviorProfile.likelyMotivators.map((motivator) => (
                    <Badge key={motivator} variant="secondary" className="whitespace-normal px-3 py-1.5">
                      {motivator}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5" />
              {detailResult.company.label} fit summary
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="leading-relaxed text-muted-foreground">{detailResult.summary}</p>
            {onOpenCompany ? (
              <Button type="button" variant="outline" onClick={() => onOpenCompany(detailResult.company.id)}>
                Open full {detailResult.company.label} analysis
              </Button>
            ) : null}

            <div className="grid gap-3 md:grid-cols-2">
              <div className="rounded-lg bg-primary/10 p-4">
                <p className="mb-2 flex items-center gap-2 font-medium text-primary">
                  <TrendingUp className="h-4 w-4" />
                  Why it fits
                </p>
                <div className="grid gap-2 text-sm text-muted-foreground">
                  {detailResult.whyFit.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              </div>
              <div className="rounded-lg bg-destructive/10 p-4">
                <p className="mb-2 flex items-center gap-2 font-medium text-destructive">
                  <ShieldAlert className="h-4 w-4" />
                  Watch-outs
                </p>
                <div className="grid gap-2 text-sm text-muted-foreground">
                  {detailResult.watchOuts.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="border-primary/20 bg-primary/5">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5" />
              Match confidence meter
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex items-end gap-3">
              <span className="font-serif text-5xl font-semibold text-primary">
                {confidenceScore}%
              </span>
              <span className="pb-2 text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                {confidenceScore >= 70 ? "High enough to explain" : "Use with caution"}
              </span>
            </div>
            <p className="text-sm leading-6 text-muted-foreground">
              Confidence estimates whether the result has enough profile detail, review volume, and evidence
              agreement to be treated as a useful direction rather than a final answer.
            </p>
            {confidenceFactors.map((item) => (
              <div key={item.label}>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <span className="font-medium">{item.label}</span>
                  <span className="text-muted-foreground">{item.value}%</span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                  <div className="h-full rounded-full bg-primary" style={{ width: `${item.value}%` }} />
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Interactive score explanation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-6 rounded-xl border border-primary/20 bg-primary/5 p-5">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <div>
                <p className="text-sm font-medium text-muted-foreground">
                  {detailResult.company.label} match score
                </p>
                <p className="mt-1 font-serif text-5xl font-semibold text-primary">
                  {detailResult.score}%
                </p>
              </div>
              <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
                This breaks the score into behavioral fit, matching themes, risk drag, and evidence confidence so
                the percentage is explainable instead of a black box.
              </p>
            </div>
          </div>
          <div className="grid gap-3 lg:grid-cols-2">
            {scoreFactors.map((factor) => {
              const isOpen = openScoreFactor === factor.label;
              return (
                <button
                  key={factor.label}
                  type="button"
                  onClick={() => setOpenScoreFactor(isOpen ? "" : factor.label)}
                  className="rounded-xl border border-border bg-background p-4 text-left transition-colors hover:border-primary/50"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between gap-3">
                        <span className="font-medium">{factor.label}</span>
                        <span className={factor.value < 0 ? "text-destructive" : "text-primary"}>
                          {factor.value > 0 ? "+" : ""}
                          {factor.value}
                        </span>
                      </div>
                      <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-muted">
                        <div
                          className={`h-full rounded-full ${factor.value < 0 ? "bg-destructive/70" : "bg-primary"}`}
                          style={{ width: `${factor.width}%` }}
                        />
                      </div>
                    </div>
                    {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </div>
                  {isOpen ? (
                    <p className="mt-3 text-sm leading-6 text-muted-foreground">{factor.note}</p>
                  ) : null}
                </button>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FileText className="h-5 w-5" />
            Review evidence
          </CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2">
          {detailResult.evidence.map((item, index) => (
            <div key={`${item.label}-${index}`} className="rounded-lg border border-border p-5">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <Badge variant={item.mode === "fulfillment" ? "secondary" : "outline"}>
                  {item.mode}
                </Badge>
                <p className="text-sm font-medium">{item.label}</p>
              </div>
              {renderEvidenceContext(item, highlightTerms)}
              {item.role || item.date ? (
                <p className="mt-3 text-xs text-muted-foreground">
                  {[item.role, item.date?.slice(0, 10)].filter(Boolean).join(" | ")}
                </p>
              ) : null}
            </div>
          ))}
        </CardContent>
      </Card>

      {isComparison && whyNotCompanies.length ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <XCircle className="h-5 w-5" />
              Why not these companies?
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-3">
            {whyNotCompanies.map((result) => (
              <div key={result.company.id} className="rounded-xl border border-border bg-background p-5">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <h3 className="font-serif text-2xl font-semibold">{result.company.label}</h3>
                  <Badge className={scoreClass(result.score)}>{result.score}%</Badge>
                </div>
                <p className="text-sm leading-6 text-muted-foreground">
                  {buildWhyNotReason(result, detailResult, profile)}
                </p>
              </div>
            ))}
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>{isComparison ? "Tradeoff map across your top matches" : "Need fit map"}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4">
          {tradeoffRows.map((row) => {
            const priority = Math.min(100, Math.max(0, row.priority));
            const topScore = row.top.company.domains[row.domain.key] || 0;
            const currentScore = row.current.company.domains[row.domain.key] || 0;
            const leaderScore = row.leader.company.domains[row.domain.key] || 0;
            return (
              <div key={row.domain.key} className="rounded-xl border border-border bg-background p-4">
                <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="font-medium">{row.domain.label}</p>
                    <p className="text-xs text-muted-foreground">Your priority: {priority}%</p>
                  </div>
                  <Badge variant="secondary">{isComparison ? `${detailResult.company.label} selected` : `${row.current.company.label} score`}</Badge>
                </div>
                <div className="grid gap-3 md:grid-cols-[180px_1fr] md:items-center">
                  <span className="text-xs font-medium text-muted-foreground">Priority weight</span>
                  <div className="h-3 overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full bg-olive" style={{ width: `${priority}%` }} />
                  </div>
                  <span className="text-xs font-medium text-muted-foreground">
                    {isComparison ? `Top match: ${row.top.company.label}` : row.current.company.label}
                  </span>
                  <div className="h-3 overflow-hidden rounded-full bg-muted">
                    <div className="h-full rounded-full bg-primary" style={{ width: `${isComparison ? topScore : currentScore}%` }} />
                  </div>
                  {isComparison ? (
                    <>
                      <span className="text-xs font-medium text-muted-foreground">
                        Current view: {row.current.company.label}
                      </span>
                      <div className="h-3 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-forest-light"
                          style={{ width: `${currentScore}%` }}
                        />
                      </div>
                    </>
                  ) : null}
                </div>
                <p className="mt-3 text-sm text-muted-foreground">
                  {isComparison
                    ? row.top.company.id === row.current.company.id
                      ? `${row.current.company.label} is both your top overall match and the selected company for this need area, with a ${currentScore} score. ${row.leader.company.id === row.current.company.id ? "It is also the strongest of the matched companies on this need." : `${row.leader.company.label} is strongest on this specific need at ${leaderScore}.`}`
                      : `Your top overall match, ${row.top.company.label}, scores ${topScore} on this need. The selected company, ${row.current.company.label}, scores ${currentScore}. ${row.leader.company.label} is strongest on this specific need at ${leaderScore}.`
                    : `${row.current.company.label} scores ${currentScore} in this need area. Compare the company score against your priority weight before trusting the overall match percentage.`}
                </p>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}

function clamp(value: number, min = 0, max = 100): number {
  return Math.min(max, Math.max(min, value));
}

function buildScoreFactors(result?: CompanyMatchResult) {
  if (!result) return [];

  const normalizedRisk = Math.round(result.riskPenalty);
  const factors = [
    {
      label: "Need fit",
      value: Math.round(result.domainAlignment),
      max: 60,
      note: "Weighted need fit compares the user's behavioral priorities against the company's scored goal profile.",
    },
    {
      label: "Theme lift",
      value: Math.round(result.themeBonus),
      max: 25,
      note: "Theme lift is added when review evidence repeatedly matches what the user says they want.",
    },
    {
      label: "Risk drag",
      value: -Math.abs(normalizedRisk),
      max: 25,
      note: "Risk drag reduces the score when the company's hindrance themes overlap with the user's stated sensitivities.",
    },
    {
      label: "Evidence confidence",
      value: Math.round(result.confidence * 20),
      max: 20,
      note: "Evidence confidence increases when the match is supported by enough review evidence instead of isolated examples.",
    },
  ];

  return factors.map((factor) => ({
    ...factor,
    width: clamp(Math.round((Math.abs(factor.value) / factor.max) * 100)),
  }));
}

function buildConfidenceFactors(profile: MatchProfile, result?: CompanyMatchResult) {
  if (!result) return [];

  const reviewVolume = clamp(Math.round(Math.log10(Math.max(10, result.company.reviews)) * 25));
  const profileClarity = clamp(Math.round((profile.behaviorProfile.confidence || 0.65) * 100));
  const evidenceAgreement = clamp(Math.round(result.confidence * 20));
  const scoreSeparation = clamp(
    Math.round(55 + Math.min(35, Math.abs(result.themeBonus) + Math.abs(result.riskPenalty)))
  );
  const dataCompleteness = clamp(
    Math.round(60 + Math.min(35, result.evidence.length * 8 + Object.keys(result.company.domains).length * 2))
  );

  return [
    { label: "Profile clarity", value: profileClarity },
    { label: "Review volume", value: reviewVolume },
    { label: "Evidence agreement", value: evidenceAgreement },
    { label: "Score separation", value: scoreSeparation },
    { label: "Data completeness", value: dataCompleteness },
  ];
}

function buildHighlightTerms(profile: MatchProfile, result?: CompanyMatchResult): string[] {
  if (!result) return [];

  const evidenceLabels = result.evidence.flatMap((item) => [item.label, item.domain]);
  return uniqueTerms([
    ...profile.desiredThemes,
    ...profile.avoidThemes,
    ...profile.behaviorProfile.coreNeeds,
    ...profile.behaviorProfile.riskSensitivities,
    ...profile.behaviorProfile.likelyMotivators,
    ...profile.matchedGoalTerms.map((term) => term.term),
    ...evidenceLabels,
  ]).filter((term) => term.length > 3);
}

function uniqueTerms(values: string[]): string[] {
  return Array.from(
    new Set(
      values
        .map((value) => value.toLowerCase().trim())
        .filter(Boolean)
    )
  ).sort((a, b) => b.length - a.length);
}

function splitSentences(text: string): string[] {
  return text
    .replace(/\s+/g, " ")
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
}

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
];

function countCueHits(sentence: string, cues: string[]): number {
  const normalized = sentence.toLowerCase();
  return cues.filter((cue) => normalized.includes(cue)).length;
}

function scoreSentence(sentence: string, terms: string[], mode: MatchEvidence["mode"]): number {
  const normalized = sentence.toLowerCase();
  const termScore = terms.reduce((score, term) => {
    if (!normalized.includes(term.toLowerCase())) return score;
    return score + Math.max(1, Math.min(4, Math.ceil(term.length / 12)));
  }, 0);

  const positiveScore = countCueHits(sentence, POSITIVE_EVIDENCE_CUES);
  const negativeScore = countCueHits(sentence, NEGATIVE_EVIDENCE_CUES);

  if (mode === "fulfillment") {
    return termScore + positiveScore * 2 - negativeScore * 3;
  }

  return termScore + negativeScore * 2 - positiveScore;
}

function selectEvidenceSentences(text: string, terms: string[], mode: MatchEvidence["mode"]): string[] {
  const sentences = splitSentences(text);
  const scored = sentences
    .map((sentence, index) => ({ sentence, index, score: scoreSentence(sentence, terms, mode) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score || a.index - b.index)
    .slice(0, 2)
    .sort((a, b) => a.index - b.index)
    .map((item) => item.sentence);

  if (scored.length) {
    return scored;
  }

  const preferredFallback = sentences.filter((sentence) =>
    mode === "fulfillment"
      ? countCueHits(sentence, NEGATIVE_EVIDENCE_CUES) === 0
      : countCueHits(sentence, NEGATIVE_EVIDENCE_CUES) > 0
  );

  return (preferredFallback.length ? preferredFallback : sentences).slice(0, 2);
}

function renderEvidenceContext(item: MatchEvidence, terms: string[]) {
  const selectedSentences = selectEvidenceSentences(item.text, terms, item.mode);
  const selectedSet = new Set(selectedSentences);
  const fullSentences = splitSentences(item.text);
  const reason =
    item.mode === "fulfillment"
      ? `This excerpt is treated as positive fit evidence for ${item.label.toLowerCase()} because it describes the workplace condition directly rather than only using a generic positive word.`
      : `This excerpt is treated as caution evidence for ${item.label.toLowerCase()} because it describes a concrete workplace friction that can reduce fit for this profile.`;

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-primary/20 bg-primary/5 p-4">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-primary">
          Highlighted context
        </p>
        <div className="space-y-2">
          {selectedSentences.map((sentence) => (
            <p key={sentence} className="text-sm leading-relaxed text-foreground">
              {sentence}
            </p>
          ))}
        </div>
        <p className="mt-3 text-xs leading-5 text-muted-foreground">{reason}</p>
      </div>
      <p className="text-sm leading-relaxed text-muted-foreground">
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
    </div>
  );
}

function buildWhyNotReason(
  result: CompanyMatchResult,
  selected: CompanyMatchResult,
  profile: MatchProfile
): string {
  const strongestNeed = MATCH_DOMAINS.reduce(
    (best, domain) => (profile.weights[domain.key] > profile.weights[best] ? domain.key : best),
    "selfprot" as MatchDomainKey
  );
  const strongestNeedLabel =
    MATCH_DOMAINS.find((domain) => domain.key === strongestNeed)?.label || "your highest-priority need";
  const selectedNeedScore = selected.company.domains[strongestNeed] || 0;
  const resultNeedScore = result.company.domains[strongestNeed] || 0;
  const scoreGap = selected.score - result.score;

  if (resultNeedScore > selectedNeedScore) {
    return `${result.company.label} is stronger on ${strongestNeedLabel.toLowerCase()}, but it loses overall because other need areas and evidence confidence do not support the profile as consistently.`;
  }

  if (scoreGap <= 3) {
    return `${result.company.label} is a close alternative. The difference is small, so the deciding factor should be the review evidence and watch-outs rather than the percentage alone.`;
  }

  const watchOut = result.watchOuts[0]?.toLowerCase();
  return `${result.company.label} trails because its evidence is weaker against ${strongestNeedLabel.toLowerCase()} and ${watchOut || "its watch-outs create more uncertainty for this profile"}.`;
}
