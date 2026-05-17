import { type CompanyComparisonMetric } from "@/lib/comparison";

export type MatchDomainKey = "phys" | "selfprot" | "aff" | "stat" | "fam";
export type MatchTradeoffKey =
  | "avoid_burnout"
  | "avoid_toxicity"
  | "growth"
  | "flexibility"
  | "belonging"
  | "stability"
  | "compensation"
  | "autonomy"
  | "manager_support"
  | "purpose"
  | "predictable_pace";

export type MatchAcceptanceKey =
  | "accept_faster_pace_for_growth"
  | "accept_lower_pay_for_flexibility"
  | "accept_less_flex_for_stability"
  | "accept_bureaucracy_for_security"
  | "accept_less_prestige_for_belonging"
  | "accept_ambiguity_for_autonomy"
  | "accept_pressure_for_pay"
  | "accept_slower_growth_for_low_politics";

export interface MatchInput {
  narrative?: string;
  tradeoffs: MatchTradeoffKey[];
  acceptedTradeoffs?: MatchAcceptanceKey[];
}

export interface MatchProfile {
  weights: Record<MatchDomainKey, number>;
  desiredThemes: string[];
  avoidThemes: string[];
  matchedGoalTerms: Array<{
    domain: MatchDomainKey;
    mode: "fulfillment" | "hindrance";
    term: string;
  }>;
  method: string;
  behaviorProfile: BehaviorProfile;
}

export interface BehaviorProfile {
  summary: string;
  likelyMotivators: string[];
  coreNeeds: string[];
  riskSensitivities: string[];
  workStyle: string;
  confidence: number;
  source: "fallback" | "llm";
}

export interface MatchEvidence {
  mode: "fulfillment" | "hindrance";
  domain: string;
  label: string;
  text: string;
  role?: string;
  date?: string;
  rating?: number;
}

export interface CompanyMatchResult {
  company: CompanyComparisonMetric;
  score: number;
  domainAlignment: number;
  themeBonus: number;
  riskPenalty: number;
  confidence: number;
  summary: string;
  whyFit: string[];
  watchOuts: string[];
  evidence: MatchEvidence[];
}

export const MATCH_DOMAINS: Array<{
  key: MatchDomainKey;
  label: string;
  short: string;
}> = [
  { key: "phys", label: "Physiological", short: "Pay" },
  { key: "selfprot", label: "Self-Protection", short: "Safety" },
  { key: "aff", label: "Affiliation", short: "Belonging" },
  { key: "stat", label: "Status & Esteem", short: "Growth" },
  { key: "fam", label: "Family Care", short: "Flex" },
];

export const MATCH_TRADEOFFS: Array<{
  key: MatchTradeoffKey;
  label: string;
  weights: Partial<Record<MatchDomainKey, number>>;
  desired?: string[];
  avoid?: string[];
}> = [
  {
    key: "avoid_burnout",
    label: "Sustainable pace",
    weights: { phys: 2, fam: 2 },
    desired: ["manageable workload", "reasonable hours", "sustainable pace"],
    avoid: ["burnout", "long hours", "unsustainable workload", "overworked"],
  },
  {
    key: "avoid_toxicity",
    label: "Low politics",
    weights: { selfprot: 3, aff: 1 },
    desired: ["fairness", "trust", "respectful"],
    avoid: ["toxic leadership", "retaliation", "favoritism", "politics"],
  },
  {
    key: "growth",
    label: "Career growth",
    weights: { stat: 3 },
    desired: ["career growth", "mentorship", "recognition"],
  },
  {
    key: "flexibility",
    label: "Flexibility",
    weights: { fam: 3 },
    desired: ["remote work", "flexible schedule", "work-life balance"],
  },
  {
    key: "belonging",
    label: "Belonging",
    weights: { aff: 3, selfprot: 1 },
    desired: ["supportive coworkers", "collaboration", "inclusive culture"],
  },
  {
    key: "stability",
    label: "Job stability",
    weights: { selfprot: 2, phys: 1 },
    desired: ["job security", "clear leadership", "reliable organization"],
  },
  {
    key: "compensation",
    label: "Pay & benefits",
    weights: { phys: 3, stat: 1 },
    desired: ["strong pay", "benefits", "bonus"],
  },
  {
    key: "autonomy",
    label: "Autonomy",
    weights: { stat: 2, selfprot: 1 },
    desired: ["autonomy", "decision-making freedom", "ownership"],
    avoid: ["micromanagement"],
  },
  {
    key: "manager_support",
    label: "Supportive manager",
    weights: { selfprot: 2, aff: 1, stat: 1 },
    desired: ["supportive management", "clear feedback", "coaching"],
    avoid: ["poor management", "lack of feedback", "retaliation"],
  },
  {
    key: "purpose",
    label: "Meaningful work",
    weights: { stat: 2, aff: 1 },
    desired: ["high impact work", "mission", "meaningful projects"],
  },
  {
    key: "predictable_pace",
    label: "Predictable schedule",
    weights: { fam: 2, phys: 1 },
    desired: ["predictable schedule", "personal time", "reasonable hours"],
    avoid: ["unpredictable hours", "always on", "late nights"],
  },
];

export const MATCH_ACCEPTANCE_TRADEOFFS: Array<{
  key: MatchAcceptanceKey;
  label: string;
  description: string;
  requires: MatchTradeoffKey[];
  weights: Partial<Record<MatchDomainKey, number>>;
  desired?: string[];
  avoid?: string[];
  tolerance?: string[];
}> = [
  {
    key: "accept_faster_pace_for_growth",
    label: "Accept faster pace for growth",
    description: "Useful when advancement matters more than a perfectly calm workload.",
    requires: ["growth"],
    weights: { stat: 1.2 },
    desired: ["career growth", "learning opportunities", "mentorship"],
    tolerance: ["fast-paced", "high standards"],
  },
  {
    key: "accept_lower_pay_for_flexibility",
    label: "Accept lower pay for flexibility",
    description: "Some workers trade compensation for schedule control or remote work.",
    requires: ["flexibility", "predictable_pace"],
    weights: { fam: 1.2 },
    desired: ["flexible schedule", "remote work", "personal time"],
    tolerance: ["lower pay"],
  },
  {
    key: "accept_less_flex_for_stability",
    label: "Accept less flexibility for stability",
    description: "Fits users who prefer predictable employment over maximum autonomy.",
    requires: ["stability"],
    weights: { selfprot: 1.2, phys: 0.6 },
    desired: ["job security", "reliable organization"],
    tolerance: ["less flexibility", "more structure"],
  },
  {
    key: "accept_bureaucracy_for_security",
    label: "Accept bureaucracy for security",
    description: "Large stable firms can bring process friction alongside security.",
    requires: ["stability", "compensation"],
    weights: { selfprot: 1, phys: 0.5 },
    desired: ["benefits", "job security", "clear leadership"],
    tolerance: ["bureaucracy", "slow decisions"],
  },
  {
    key: "accept_less_prestige_for_belonging",
    label: "Accept less prestige for belonging",
    description: "Prioritizes daily team climate over employer brand status.",
    requires: ["belonging", "manager_support"],
    weights: { aff: 1.2, selfprot: 0.4 },
    desired: ["supportive coworkers", "inclusive culture"],
    tolerance: ["less prestige"],
  },
  {
    key: "accept_ambiguity_for_autonomy",
    label: "Accept ambiguity for autonomy",
    description: "Autonomous roles can come with less structure and clearer ownership pressure.",
    requires: ["autonomy"],
    weights: { stat: 1, selfprot: 0.3 },
    desired: ["autonomy", "ownership", "decision-making freedom"],
    tolerance: ["ambiguity", "less structure"],
  },
  {
    key: "accept_pressure_for_pay",
    label: "Accept pressure for pay",
    description: "Compensation-focused choices sometimes involve higher intensity.",
    requires: ["compensation"],
    weights: { phys: 1.2, stat: 0.5 },
    desired: ["strong pay", "bonus", "benefits"],
    tolerance: ["high pressure", "long hours"],
  },
  {
    key: "accept_slower_growth_for_low_politics",
    label: "Accept slower growth for low politics",
    description: "A calmer, fairer culture may matter more than rapid promotion velocity.",
    requires: ["avoid_toxicity"],
    weights: { selfprot: 1.2, aff: 0.5 },
    desired: ["fairness", "trust", "respectful"],
    tolerance: ["slower growth"],
  },
];

const GOAL_DICTIONARY: Record<
  MatchDomainKey,
  {
    fulfillment: string[];
    hindrance: string[];
  }
> = {
  phys: {
    fulfillment: [
      "good pay",
      "fair compensation",
      "bonus",
      "benefits",
      "healthcare",
      "reasonable hours",
      "manageable workload",
      "sustainable pace",
      "stability",
      "time off",
    ],
    hindrance: [
      "low pay",
      "underpaid",
      "overworked",
      "long hours",
      "unpaid overtime",
      "no benefits",
      "burnout",
      "stressful",
      "understaffed",
      "weekend work",
    ],
  },
  selfprot: {
    fulfillment: [
      "safe",
      "fairness",
      "fair management",
      "fair managers",
      "respectful",
      "trust",
      "ethical",
      "job security",
      "accountability",
      "management cares",
      "supportive manager",
      "supportive managers",
      "boundaries honored",
      "equal treatment",
    ],
    hindrance: [
      "toxic",
      "unsafe",
      "harassment",
      "bullying",
      "retaliation",
      "unethical",
      "favoritism",
      "politics",
      "unfair",
      "glass ceiling",
    ],
  },
  aff: {
    fulfillment: [
      "teamwork",
      "collaboration",
      "friendly",
      "belonging",
      "inclusive",
      "supportive",
      "colleagues",
      "open communication",
      "community",
      "positive culture",
    ],
    hindrance: [
      "isolation",
      "disconnected",
      "unfriendly",
      "cliques",
      "competition",
      "no support",
      "gossip",
      "poor communication",
      "siloed",
      "hostile environment",
    ],
  },
  stat: {
    fulfillment: [
      "promotion",
      "recognition",
      "valued",
      "advancement",
      "growth",
      "career path",
      "career growth",
      "mentorship",
      "learning opportunities",
      "high impact work",
    ],
    hindrance: [
      "unrecognized",
      "overlooked",
      "no growth",
      "dead end",
      "micromanagement",
      "lack of feedback",
      "limited advancement",
      "no visibility",
      "limited learning",
      "stagnation",
    ],
  },
  fam: {
    fulfillment: [
      "work life balance",
      "family friendly",
      "parental leave",
      "childcare",
      "flexibility",
      "flexible hours",
      "work from home",
      "remote work",
      "predictable schedule",
      "personal time",
    ],
    hindrance: [
      "no parental leave",
      "inflexible schedule",
      "limited time off",
      "unpredictable hours",
      "long hours",
      "no work life balance",
      "weekend work",
      "late nights",
      "always on",
      "no flexibility",
    ],
  },
};

function normalizeText(value: string): string {
  return value.toLowerCase().replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();
}

function emptyWeights(): Record<MatchDomainKey, number> {
  return { phys: 1, selfprot: 1, aff: 1, stat: 1, fam: 1 };
}

function normalizeWeights(raw: Record<MatchDomainKey, number>): Record<MatchDomainKey, number> {
  const total = Object.values(raw).reduce((sum, value) => sum + value, 0) || 1;
  return Object.fromEntries(
    MATCH_DOMAINS.map((domain) => [domain.key, raw[domain.key] / total])
  ) as Record<MatchDomainKey, number>;
}

function hasTheme(text: string, theme: string): boolean {
  return text.includes(normalizeText(theme));
}

function humanizeTerm(term: string): string {
  return term.replace(/_/g, " ");
}

function unique(values: string[]): string[] {
  return Array.from(new Set(values.filter(Boolean)));
}

const GENERIC_MATCH_WORDS = new Set([
  "i",
  "me",
  "my",
  "need",
  "needs",
  "want",
  "wants",
  "work",
  "job",
  "company",
  "place",
  "good",
  "great",
  "nice",
  "best",
  "employee",
  "employer",
]);

function meaningfulTokens(text: string): string[] {
  return normalizeText(text)
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 4 && !GENERIC_MATCH_WORDS.has(token));
}

function countGoalTermHits(text: string): number {
  if (!text) return 0;
  return MATCH_DOMAINS.reduce((sum, domain) => {
    const dictionary = GOAL_DICTIONARY[domain.key];
    const terms = [...dictionary.fulfillment, ...dictionary.hindrance];
    return sum + terms.filter((term) => hasTheme(text, term)).length;
  }, 0);
}

export function validateMatchInput(input: MatchInput): string | null {
  const narrative = normalizeText(input.narrative || "");
  const selectedCount = input.tradeoffs.length + (input.acceptedTradeoffs?.length || 0);
  if (!narrative) {
    return "Describe what you want at work. The buttons help structure the match, but I still need your own workplace context.";
  }

  const tokens = meaningfulTokens(narrative);
  const goalHits = countGoalTermHits(narrative);
  if (tokens.length < 3 || goalHits === 0) {
    return selectedCount > 0
      ? "The selected buttons are useful, but the text still needs workplace-relevant detail. Add a sentence about what you want, such as growth, flexibility, fair management, pay, stability, belonging, or autonomy."
      : "I need a little more workplace-relevant detail before matching. Add goals like growth, flexibility, fair management, pay, stability, belonging, or autonomy.";
  }
  return null;
}

export function buildMatchProfile(input: MatchInput): MatchProfile {
  const text = normalizeText(input.narrative || "");
  const weights = emptyWeights();
  const desiredThemes: string[] = [];
  const avoidThemes: string[] = [];
  const matchedGoalTerms: MatchProfile["matchedGoalTerms"] = [];

  MATCH_DOMAINS.forEach((domain) => {
    GOAL_DICTIONARY[domain.key].fulfillment.forEach((term) => {
      if (text && hasTheme(text, term)) {
        weights[domain.key] += 1.2;
        desiredThemes.push(humanizeTerm(term));
        matchedGoalTerms.push({ domain: domain.key, mode: "fulfillment", term: humanizeTerm(term) });
      }
    });

    GOAL_DICTIONARY[domain.key].hindrance.forEach((term) => {
      if (text && hasTheme(text, term)) {
        weights[domain.key] += 1.5;
        avoidThemes.push(humanizeTerm(term));
        matchedGoalTerms.push({ domain: domain.key, mode: "hindrance", term: humanizeTerm(term) });
      }
    });
  });

  input.tradeoffs.forEach((tradeoffKey) => {
    const tradeoff = MATCH_TRADEOFFS.find((item) => item.key === tradeoffKey);
    if (!tradeoff) return;

    Object.entries(tradeoff.weights).forEach(([key, value]) => {
      weights[key as MatchDomainKey] += value || 0;
    });
    tradeoff.desired?.forEach((theme) => desiredThemes.push(theme));
    tradeoff.avoid?.forEach((theme) => avoidThemes.push(theme));
  });

  input.acceptedTradeoffs?.forEach((acceptanceKey) => {
    const acceptance = MATCH_ACCEPTANCE_TRADEOFFS.find((item) => item.key === acceptanceKey);
    if (!acceptance) return;

    Object.entries(acceptance.weights).forEach(([key, value]) => {
      weights[key as MatchDomainKey] += value || 0;
    });
    acceptance.desired?.forEach((theme) => desiredThemes.push(theme));
    acceptance.avoid?.forEach((theme) => avoidThemes.push(theme));
    acceptance.tolerance?.forEach((theme) => desiredThemes.push(`tolerates ${theme}`));
  });

  const normalizedWeights = normalizeWeights(weights);
  const topDomains = [...MATCH_DOMAINS].sort(
    (a, b) => normalizedWeights[b.key] - normalizedWeights[a.key]
  );
  const topNeed = topDomains[0];
  const secondNeed = topDomains[1];
  const selectedLabels = [
    ...input.tradeoffs.map((key) => MATCH_TRADEOFFS.find((item) => item.key === key)?.label),
    ...(input.acceptedTradeoffs || []).map((key) => MATCH_ACCEPTANCE_TRADEOFFS.find((item) => item.key === key)?.label),
  ]
    .filter(Boolean) as string[];

  const fallbackBehavior: BehaviorProfile = {
    summary: `Your profile suggests you are looking for a workplace that protects ${topNeed.label.toLowerCase()} while still supporting ${secondNeed.label.toLowerCase()}. The strongest signals come from your selected tradeoffs and any goal-dictionary language in your description.`,
    likelyMotivators: unique([
      ...desiredThemes,
      ...selectedLabels.filter((label) => !label.toLowerCase().includes("low politics")),
    ]).slice(0, 5),
    coreNeeds: topDomains.slice(0, 3).map((domain) => domain.label),
    riskSensitivities: unique([
      ...avoidThemes,
      ...selectedLabels.filter((label) => label.toLowerCase().includes("low politics")),
    ]).slice(0, 5),
    workStyle:
      "You are likely to evaluate employers through the daily conditions they create, not just through brand prestige or generic ratings.",
    confidence: input.narrative?.trim() ? 0.74 : 0.56,
    source: "fallback",
  };

  return {
    weights: normalizedWeights,
    desiredThemes: unique(desiredThemes).slice(0, 10),
    avoidThemes: unique(avoidThemes).slice(0, 10),
    matchedGoalTerms,
    method: "Goal profile",
    behaviorProfile: fallbackBehavior,
  };
}

export function mergeBehaviorProfile(profile: MatchProfile, behaviorProfile?: Partial<BehaviorProfile>): MatchProfile {
  if (!behaviorProfile) {
    return profile;
  }

  return {
    ...profile,
    behaviorProfile: {
      ...profile.behaviorProfile,
      ...behaviorProfile,
      source: behaviorProfile.source || "llm",
    },
  };
}

export function cleanReviewText(value: string): string {
  const replacements: Array<[RegExp, string]> = [
    [/\u00e2\u0080\u0099/g, "'"],
    [/\u00e2\u0080\u0098/g, "'"],
    [/\u00e2\u0080\u009c/g, '"'],
    [/\u00e2\u0080\u009d/g, '"'],
    [/\u00e2\u0080\u0094/g, "-"],
    [/\u00e2\u0080\u0093/g, "-"],
    [/\u00e2\u0080\u00a2/g, " "],
    [/â€™/g, "'"],
    [/â€˜/g, "'"],
    [/â€œ/g, '"'],
    [/â€/g, '"'],
    [/â/g, '"'],
    [/â/g, '"'],
    [/â€”/g, "-"],
    [/â€“/g, "-"],
    [/â/g, "-"],
    [/â/g, "-"],
    [/â€¢/g, " "],
    [/Â/g, ""],
    [/�/g, ""],
  ];

  let cleaned = value || "";
  replacements.forEach(([pattern, replacement]) => {
    cleaned = cleaned.replace(pattern, replacement);
  });
  return cleaned
    .replace(/[*#_`~]+/g, " ")
    .replace(/["“”]+/g, "")
    .replace(/[•·]+/g, " ")
    .replace(/\p{Cc}+/gu, " ")
    .replace(/\.\.\.$/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function topicText(topic: CompanyComparisonMetric["topics"][number]): string {
  return normalizeText(`${topic.label} ${topic.domain} ${topic.terms.join(" ")}`);
}

function matchThemes(themes: string[], topic: CompanyComparisonMetric["topics"][number]): boolean {
  const text = topicText(topic);
  return themes.some((theme) => {
    const normalized = normalizeText(theme);
    return normalized.length > 3 && text.includes(normalized);
  });
}

function parseRating(value: unknown): number | undefined {
  const rating = Number(value);
  return Number.isFinite(rating) ? rating : undefined;
}

function isUnsuitableEvidence(
  mode: "fulfillment" | "hindrance",
  text: string,
  rating: number | undefined,
  targetThemes: string[]
): boolean {
  if (mode === "fulfillment") {
    return rating !== undefined && rating <= 2;
  }

  if (rating === undefined || rating < 4) {
    return false;
  }

  const normalizedText = normalizeText(text);
  const hasDirectRisk = targetThemes.some(
    (theme) => theme.length > 3 && normalizedText.includes(theme)
  );
  return !hasDirectRisk;
}

function companyDomainScore(company: CompanyComparisonMetric, key: MatchDomainKey): number {
  return Number(company.domains[key] || 50);
}

function extractEvidence(rag: unknown, profile: MatchProfile): MatchEvidence[] {
  const raw = rag && typeof rag === "object" ? (rag as Record<string, unknown>) : {};
  const evidencePayload = raw.evidence && typeof raw.evidence === "object"
    ? (raw.evidence as Record<string, unknown>)
    : raw;
  const clusters = Array.isArray(evidencePayload.clusters) ? evidencePayload.clusters : [];
  const desired = profile.desiredThemes.map(normalizeText);
  const avoided = profile.avoidThemes.map(normalizeText);

  return clusters
    .flatMap((cluster) => {
      const row = cluster && typeof cluster === "object" ? (cluster as Record<string, unknown>) : {};
      const mode = row.mode === "hindrance" ? "hindrance" : "fulfillment";
      const label = String(row.label || "Review evidence");
      const domain = String(row.domain || "");
      const terms = Array.isArray(row.terms) ? row.terms.map((term) => normalizeText(String(term))) : [];
      const targetThemes = mode === "hindrance" ? avoided : desired;
      const relevant = targetThemes.length === 0 || targetThemes.some((theme) =>
        terms.some((term) => term.includes(theme) || theme.includes(term))
      );

      if (!relevant || !Array.isArray(row.evidence)) {
        return [];
      }

      return row.evidence
        .map((item) => {
          const evidence = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
          const text = cleanReviewText(String(evidence.text || ""));
          const rating = parseRating(evidence.rating);
          if (!text || isUnsuitableEvidence(mode, text, rating, targetThemes)) {
            return null;
          }
          return {
            mode,
            domain,
            label,
            text,
            role: typeof evidence.role === "string" ? evidence.role : undefined,
            date: typeof evidence.date === "string" ? evidence.date : undefined,
            rating,
          };
        })
        .filter((item): item is MatchEvidence => Boolean(item))
        .slice(0, 2);
    })
    .filter((item) => item.text)
    .slice(0, 5);
}

function fallbackEvidence(company: CompanyComparisonMetric, profile: MatchProfile): MatchEvidence[] {
  const desired = company.topics
    .filter((topic) => topic.mode === "fulfillment" && matchThemes(profile.desiredThemes, topic))
    .slice(0, 2)
    .map((topic) => ({
      mode: "fulfillment" as const,
      domain: topic.domain,
      label: topic.label,
      text: cleanReviewText(`Review cluster evidence: ${topic.terms.slice(0, 5).join(", ")}.`),
    }));

  const risks = company.topics
    .filter((topic) => topic.mode === "hindrance" && matchThemes(profile.avoidThemes, topic))
    .slice(0, 2)
    .map((topic) => ({
      mode: "hindrance" as const,
      domain: topic.domain,
      label: topic.label,
      text: cleanReviewText(`Risk cluster evidence: ${topic.terms.slice(0, 5).join(", ")}.`),
    }));

  return [...desired, ...risks].slice(0, 4);
}

export function scoreCompanyMatch(
  profile: MatchProfile,
  company: CompanyComparisonMetric,
  rag?: unknown
): CompanyMatchResult {
  const domainAlignment = MATCH_DOMAINS.reduce(
    (sum, domain) => sum + companyDomainScore(company, domain.key) * profile.weights[domain.key],
    0
  );

  const fulfillmentHits = company.topics.filter(
    (topic) => topic.mode === "fulfillment" && matchThemes(profile.desiredThemes, topic)
  );
  const riskHits = company.topics.filter(
    (topic) => topic.mode === "hindrance" && matchThemes(profile.avoidThemes, topic)
  );

  const themeBonus = Math.min(
    12,
    fulfillmentHits.reduce((sum, topic) => sum + Math.max(2, topic.signal / 10), 0)
  );
  const riskPenalty = Math.min(
    24,
    riskHits.reduce((sum, topic) => sum + Math.max(3, topic.signal / 5), 0)
  );
  const confidence = Math.min(5, Math.log10(Math.max(10, company.reviews)) * 1.4);
  const score = Math.round(
    Math.max(0, Math.min(100, domainAlignment + themeBonus + confidence - riskPenalty))
  );

  const strongestNeed = MATCH_DOMAINS.reduce((best, domain) =>
    profile.weights[domain.key] > profile.weights[best.key] ? domain : best
  );
  const bestCompanyDomain = MATCH_DOMAINS.reduce((best, domain) =>
    companyDomainScore(company, domain.key) > companyDomainScore(company, best.key) ? domain : best
  );
  const evidence = extractEvidence(rag, profile);
  const visibleEvidence = evidence.length ? evidence : fallbackEvidence(company, profile);

  const whyFit = [
    `${company.label} is strongest around ${bestCompanyDomain.label.toLowerCase()}, while your profile weights ${strongestNeed.label.toLowerCase()} most heavily.`,
    ...fulfillmentHits.slice(0, 2).map((topic) =>
      `${topic.label} appears as a positive review cluster with terms like ${topic.terms.slice(0, 3).join(", ")}.`
    ),
  ].slice(0, 3);

  const watchOuts = riskHits.length
    ? riskHits.slice(0, 3).map((topic) =>
        `${topic.label} may matter because your profile flags ${profile.avoidThemes.slice(0, 2).join(" and ")} as sensitive risks.`
      )
    : [`No major matched risk cluster was found for the selected tradeoffs, but team and manager variation can still matter.`];

  return {
    company,
    score,
    domainAlignment: Math.round(domainAlignment),
    themeBonus: Math.round(themeBonus),
    riskPenalty: Math.round(riskPenalty),
    confidence: Math.round(confidence),
    summary: `${company.label} scores ${score}% against this behavioral profile. The score combines goal-domain alignment, goal-dictionary theme matches, risk penalties, and review-evidence confidence.`,
    whyFit,
    watchOuts,
    evidence: visibleEvidence,
  };
}

export function scoreClass(score: number): string {
  if (score >= 80) return "bg-primary text-primary-foreground";
  if (score >= 70) return "bg-olive-muted/70 text-foreground";
  return "bg-destructive/15 text-destructive";
}
