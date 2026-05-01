import { getJobDownloadUrl } from "@/lib/backend-api";

const GOAL_ORDER = ["phys", "selfprot", "aff", "stat", "fam"];

const GOAL_LABELS: Record<string, string> = {
  phys: "Physiological Needs",
  selfprot: "Self-Protection",
  aff: "Affiliation",
  stat: "Status & Esteem",
  fam: "Family Care",
};

const GOAL_SHORT: Record<string, string> = {
  phys: "Phys",
  selfprot: "Prot",
  aff: "Affil",
  stat: "Status",
  fam: "Family",
};

export interface DomainScore {
  code: string;
  domain: string;
  short: string;
  fulfillment: number;
  hindrance: number;
  fulfillmentPct: number;
  hindrancePct: number;
  scorePct: number;
}

export interface AnalysisDownloads {
  cleanedReviews?: string;
  reviewScores: string;
  companyScores: string;
  topicSummary?: string;
  topicAssignments?: string;
}

export interface TopicCluster {
  mode: "fulfillment" | "hindrance";
  clusterId: string;
  label: string;
  domain: string;
  goal: string;
  reviewCount: number;
  signal: number;
  x: number;
  y: number;
  terms: string[];
  summary?: string;
  evidenceIds?: string[];
}

export interface RagSummary {
  executiveSummary: string[];
  keyStrengths: string[];
  keyRisks: string[];
  domainExplanations: {
    strongest?: string;
    weakest?: string;
  };
  model?: string;
  generatedAt?: string;
}

export interface RagInsights {
  whatStandsOut: string[];
  model?: string;
  generatedAt?: string;
}

export interface RagArtifacts {
  summary?: RagSummary;
  insights?: RagInsights;
}

export interface AnalysisResult {
  jobId: string;
  companyName: string;
  companyId: string;
  reviewCount: number;
  overallScore: number;
  strongestDomain: DomainScore;
  weakestDomain: DomainScore;
  domainScores: DomainScore[];
  topicClusters: TopicCluster[];
  rag?: RagArtifacts;
  analysisDate: string;
  downloads: AnalysisDownloads;
}

interface BuildArgs {
  jobId: string;
  inputCompanyName: string;
  resolvedCompanyName?: string;
  reviewCsvText: string;
  companyCsvText: string;
  topicCsvText?: string;
  ragSummary?: unknown;
  ragClusters?: unknown;
  ragInsights?: unknown;
  outputFiles?: string[];
  downloads?: AnalysisDownloads;
}

type CsvRow = Record<string, string>;

function normalizeCompanyKey(raw: string): string {
  return (raw || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

function parseCsv(text: string): CsvRow[] {
  const rows: string[][] = [];
  const data = text.replace(/^\uFEFF/, "");
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < data.length; i += 1) {
    const ch = data[i];

    if (inQuotes) {
      if (ch === '"') {
        if (data[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
      continue;
    }

    if (ch === ",") {
      row.push(field);
      field = "";
      continue;
    }

    if (ch === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }

    if (ch !== "\r") {
      field += ch;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  if (rows.length < 2) {
    return [];
  }

  const header = rows[0];
  const out: CsvRow[] = [];
  for (let i = 1; i < rows.length; i += 1) {
    const rec = rows[i];
    if (rec.length === 1 && rec[0] === "") {
      continue;
    }
    const rowObj: CsvRow = {};
    for (let c = 0; c < header.length; c += 1) {
      rowObj[header[c]] = rec[c] ?? "";
    }
    out.push(rowObj);
  }
  return out;
}

function asNumber(value: string | undefined): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function mean(values: number[]): number {
  if (values.length === 0) {
    return 0;
  }
  const total = values.reduce((acc, cur) => acc + cur, 0);
  return total / values.length;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function detectGoalCodes(reviewRows: CsvRow[]): string[] {
  if (reviewRows.length === 0) {
    return GOAL_ORDER;
  }
  const cols = Object.keys(reviewRows[0]);
  const detected = cols
    .filter((c) => c.startsWith("F_raw_"))
    .map((c) => c.replace("F_raw_", ""));

  const uniqueDetected = Array.from(new Set(detected));
  const ordered = [
    ...GOAL_ORDER.filter((g) => uniqueDetected.includes(g)),
    ...uniqueDetected.filter((g) => !GOAL_ORDER.includes(g)),
  ];
  return ordered.length > 0 ? ordered : GOAL_ORDER;
}

function selectCompanyId(
  reviewRows: CsvRow[],
  companyRows: CsvRow[],
  inputCompanyName: string,
  resolvedCompanyName?: string
): string {
  const candidates = Array.from(
    new Set(
      companyRows
        .map((r) => (r.company_id || "").trim())
        .filter((v) => v.length > 0)
    )
  );

  if (candidates.length === 0) {
    const reviewCandidates = Array.from(
      new Set(
        reviewRows
          .map((r) => (r.company_id || "").trim())
          .filter((v) => v.length > 0)
      )
    );
    if (reviewCandidates.length === 1) {
      return reviewCandidates[0];
    }
    return inputCompanyName.trim();
  }

  const byKey = new Map<string, string>();
  for (const c of candidates) {
    byKey.set(normalizeCompanyKey(c), c);
  }

  const resolvedKey = normalizeCompanyKey(resolvedCompanyName || "");
  if (resolvedKey && byKey.has(resolvedKey)) {
    return byKey.get(resolvedKey)!;
  }

  const inputKey = normalizeCompanyKey(inputCompanyName);
  if (inputKey && byKey.has(inputKey)) {
    return byKey.get(inputKey)!;
  }

  if (candidates.length === 1) {
    return candidates[0];
  }

  return candidates[0];
}

function findCompanyRow(companyRows: CsvRow[], companyId: string): CsvRow | undefined {
  const wanted = normalizeCompanyKey(companyId);
  return companyRows.find((r) => normalizeCompanyKey(r.company_id || "") === wanted);
}

function selectCleanedPath(outputFiles: string[]): string | undefined {
  const cleaned = outputFiles.find((p) => /^02_clean\/reviews_.*\.csv$/i.test(p));
  return cleaned;
}

function parseTopicClusters(text?: string): TopicCluster[] {
  if (!text?.trim()) {
    return [];
  }
  return parseCsv(text)
    .map((row) => {
      const mode = row.mode === "hindrance" ? "hindrance" : "fulfillment";
      return {
        mode,
        clusterId: row.cluster_id || "",
        label: row.label || "Topic Cluster",
        domain: row.domain || "",
        goal: row.goal || "",
        reviewCount: asNumber(row.review_count),
        signal: asNumber(row.signal),
        x: asNumber(row.x),
        y: asNumber(row.y),
        terms: (row.terms || "")
          .split("|")
          .map((term) => term.trim())
          .filter(Boolean),
      };
    })
    .filter((cluster) => cluster.clusterId);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.map((item) => String(item || "")).filter(Boolean) : [];
}

function parseRagSummary(value: unknown): RagSummary | undefined {
  const raw = asRecord(value);
  const executiveSummary = asStringArray(raw.executive_summary);
  if (executiveSummary.length === 0) {
    return undefined;
  }
  const domain = asRecord(raw.domain_explanations);
  return {
    executiveSummary,
    keyStrengths: asStringArray(raw.key_strengths),
    keyRisks: asStringArray(raw.key_risks),
    domainExplanations: {
      strongest: typeof domain.strongest === "string" ? domain.strongest : undefined,
      weakest: typeof domain.weakest === "string" ? domain.weakest : undefined,
    },
    model: typeof raw.model === "string" ? raw.model : undefined,
    generatedAt: typeof raw.generated_at === "string" ? raw.generated_at : undefined,
  };
}

function parseRagInsights(value: unknown): RagInsights | undefined {
  const raw = asRecord(value);
  const whatStandsOut = asStringArray(raw.what_stands_out);
  if (whatStandsOut.length === 0) {
    return undefined;
  }
  return {
    whatStandsOut,
    model: typeof raw.model === "string" ? raw.model : undefined,
    generatedAt: typeof raw.generated_at === "string" ? raw.generated_at : undefined,
  };
}

function enrichTopicClusters(clusters: TopicCluster[], ragClusters: unknown): TopicCluster[] {
  const raw = asRecord(ragClusters);
  const summaries = Array.isArray(raw.cluster_summaries) ? raw.cluster_summaries : [];
  const byId = new Map<string, { summary?: string; evidenceIds?: string[] }>();
  for (const item of summaries) {
    const row = asRecord(item);
    const clusterId = typeof row.cluster_id === "string" ? row.cluster_id : "";
    if (!clusterId) {
      continue;
    }
    byId.set(clusterId, {
      summary: typeof row.summary === "string" ? row.summary : undefined,
      evidenceIds: asStringArray(row.evidence_ids),
    });
  }
  return clusters.map((cluster) => ({
    ...cluster,
    summary: byId.get(cluster.clusterId)?.summary,
    evidenceIds: byId.get(cluster.clusterId)?.evidenceIds,
  }));
}

export function buildAnalysisResult(args: BuildArgs): AnalysisResult {
  const reviewRows = parseCsv(args.reviewCsvText);
  const companyRows = parseCsv(args.companyCsvText);
  const companyId = selectCompanyId(
    reviewRows,
    companyRows,
    args.inputCompanyName,
    args.resolvedCompanyName
  );

  const companyKey = normalizeCompanyKey(companyId);
  const companyReviewRows = reviewRows.filter(
    (r) => normalizeCompanyKey(r.company_id || "") === companyKey
  );
  const activeReviewRows = companyReviewRows.length > 0 ? companyReviewRows : reviewRows;
  const companyRow = findCompanyRow(companyRows, companyId);

  const goalCodes = detectGoalCodes(activeReviewRows);
  const domainScores: DomainScore[] = goalCodes.map((code) => {
    const fVals = activeReviewRows.map((r) => asNumber(r[`F_raw_${code}`]));
    const hVals = activeReviewRows.map((r) => asNumber(r[`H_raw_${code}`]));

    const fulfillment = round2(mean(fVals));
    const hindrance = round2(mean(hVals));
    const totalEvidence = fulfillment + hindrance;
    const fulfillmentPct = totalEvidence > 0 ? round2((fulfillment / totalEvidence) * 100) : 50;
    const hindrancePct = round2(100 - fulfillmentPct);

    let goalScore = Number.NaN;
    if (companyRow) {
      goalScore = Number(companyRow[`G_smoothed_final_${code}`]);
    }
    if (!Number.isFinite(goalScore)) {
      goalScore = mean(activeReviewRows.map((r) => asNumber(r[`G_final_${code}`])));
    }

    const scorePct = round2(clamp(((goalScore + 1) / 2) * 100, 0, 100));
    return {
      code,
      domain: GOAL_LABELS[code] || code,
      short: GOAL_SHORT[code] || code,
      fulfillment,
      hindrance,
      fulfillmentPct,
      hindrancePct,
      scorePct,
    };
  });

  const nonEmptyScores = domainScores.length > 0 ? domainScores : [
    {
      code: "phys",
      domain: "Physiological Needs",
      short: "Phys",
      fulfillment: 0,
      hindrance: 0,
      fulfillmentPct: 50,
      hindrancePct: 50,
      scorePct: 50,
    },
  ];

  const strongestDomain = nonEmptyScores.reduce((a, b) => (a.scorePct >= b.scorePct ? a : b));
  const weakestDomain = nonEmptyScores.reduce((a, b) => (a.scorePct <= b.scorePct ? a : b));
  const overallScore = Math.round(mean(nonEmptyScores.map((s) => s.scorePct)));

  const resolvedName = (args.resolvedCompanyName || "").trim();
  const displayCompanyName = resolvedName || companyId || args.inputCompanyName;

  const reviewCount = activeReviewRows.length;
  const outputFiles = args.outputFiles || [];
  const cleanedPath = selectCleanedPath(outputFiles);
  const hasTopicSummary = outputFiles.includes("topic_summary.csv");
  const hasTopicAssignments = outputFiles.includes("topic_assignments.csv");

  const topicClusters = enrichTopicClusters(
    parseTopicClusters(args.topicCsvText),
    args.ragClusters
  );
  const ragSummary = parseRagSummary(args.ragSummary);
  const ragInsights = parseRagInsights(args.ragInsights);

  return {
    jobId: args.jobId,
    companyName: displayCompanyName,
    companyId,
    reviewCount,
    overallScore,
    strongestDomain,
    weakestDomain,
    domainScores: nonEmptyScores,
    topicClusters,
    rag: ragSummary || ragInsights ? { summary: ragSummary, insights: ragInsights } : undefined,
    analysisDate: new Date().toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    }),
    downloads: args.downloads || {
      cleanedReviews: cleanedPath ? getJobDownloadUrl(args.jobId, cleanedPath) : undefined,
      reviewScores: getJobDownloadUrl(args.jobId, "04_score/review_scores.csv"),
      companyScores: getJobDownloadUrl(args.jobId, "04_score/company_scores.csv"),
      topicSummary: hasTopicSummary ? getJobDownloadUrl(args.jobId, "topic_summary.csv") : undefined,
      topicAssignments: hasTopicAssignments ? getJobDownloadUrl(args.jobId, "topic_assignments.csv") : undefined,
    },
  };
}
