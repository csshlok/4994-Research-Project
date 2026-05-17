import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  MATCH_DOMAINS,
  scoreClass,
  type CompanyMatchResult,
  type MatchProfile,
} from "@/lib/matching";
import { type MatchTopSummary } from "@/lib/backend-api";
import { BarChart3, FileText, ShieldAlert, Sparkles, TrendingUp } from "lucide-react";
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
  const [selectedId, setSelectedId] = useState<string>("");
  const selectedResult =
    selectedId ? visibleResults.find((result) => result.company.id === selectedId) : undefined;
  const detailResult = selectedResult || visibleResults[0];

  useEffect(() => {
    setSelectedId("");
  }, [visibleResults[0]?.company.id]);

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

        <div className="grid gap-4 md:grid-cols-4">
          {visibleResults.map((result) => (
            <Card
              key={result.company.id}
              onClick={() => setSelectedId(result.company.id)}
              className={`cursor-pointer border-primary/20 transition-all ${
                selectedResult?.company.id === result.company.id
                  ? "border-primary shadow-elevated"
                  : "hover:border-primary/40"
              }`}
            >
              <CardContent className="pt-5">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <h2 className="font-medium">{result.company.label}</h2>
                  <Badge className={scoreClass(result.score)}>{result.score}%</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  {result.company.reviews.toLocaleString()} scored reviews
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

        <Card>
          <CardHeader>
            <CardTitle>Match signal snapshot</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            {[
              { label: "Weighted need fit", value: detailResult.domainAlignment, tone: "bg-primary" },
              { label: "Theme lift", value: detailResult.themeBonus, tone: "bg-olive" },
              { label: "Risk drag", value: detailResult.riskPenalty, tone: "bg-destructive/70" },
              { label: "Evidence confidence", value: detailResult.confidence * 20, tone: "bg-forest-light" },
            ].map((item) => (
              <div key={item.label}>
                <div className="mb-1 flex items-center justify-between text-sm">
                  <span className="font-medium">{item.label}</span>
                  <span className="text-muted-foreground">{Math.round(item.value)}</span>
                </div>
                <div className="h-2.5 overflow-hidden rounded-full bg-muted">
                  <div className={`h-full rounded-full ${item.tone}`} style={{ width: `${Math.min(100, Math.max(0, item.value))}%` }} />
                </div>
              </div>
            ))}
            <p className="text-sm leading-6 text-muted-foreground">
              This snapshot shows how the selected company earns its score: need fit adds the base,
              matching positive themes lift it, and matched risk clusters drag it down.
            </p>
          </CardContent>
        </Card>
      </section>

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
              <p className="text-sm leading-relaxed text-muted-foreground">{item.text}</p>
              {item.role || item.date ? (
                <p className="mt-3 text-xs text-muted-foreground">
                  {[item.role, item.date?.slice(0, 10)].filter(Boolean).join(" | ")}
                </p>
              ) : null}
            </div>
          ))}
        </CardContent>
      </Card>

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
