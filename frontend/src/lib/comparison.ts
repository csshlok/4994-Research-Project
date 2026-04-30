import { COMPANY_OPTIONS } from "@/lib/company-options";

export type CsvRow = Record<string, string>;

export interface CompareTopic {
  mode: string;
  label: string;
  domain: string;
  reviewCount: number;
  signal: number;
  x: number;
  y: number;
  terms: string[];
}

export interface CompanyComparisonMetric {
  id: string;
  label: string;
  overall: number;
  reviews: number;
  domains: Record<string, number>;
  topics: CompareTopic[];
}

export const COMPARISON_DOMAINS = [
  { key: "phys", label: "Physiological" },
  { key: "selfprot", label: "Self-Protection" },
  { key: "aff", label: "Affiliation" },
  { key: "stat", label: "Status & Esteem" },
  { key: "fam", label: "Family Care" },
];

export function parseCsv(text: string): CsvRow[] {
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

  if (rows.length < 2) return [];
  const header = rows[0];
  return rows.slice(1).filter((r) => r.length > 1 || r[0]).map((r) => {
    const out: CsvRow = {};
    header.forEach((h, i) => {
      out[h] = r[i] ?? "";
    });
    return out;
  });
}

function num(value: string | undefined): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : 0;
}

function scorePct(raw: number): number {
  return Math.round(Math.max(0, Math.min(100, ((raw + 1) / 2) * 100)));
}

export function displayCompanyLabel(id: string): string {
  return COMPANY_OPTIONS.find((c) => c.value.toLowerCase() === id.toLowerCase())?.label || id;
}

export function buildCompanyComparisonMetric(
  id: string,
  companyCsvText: string,
  topicCsvText: string
): CompanyComparisonMetric {
  const companyRow = parseCsv(companyCsvText)[0] || {};
  const domains = Object.fromEntries(
    COMPARISON_DOMAINS.map((domain) => [
      domain.key,
      scorePct(num(companyRow[`G_smoothed_final_${domain.key}`])),
    ])
  );
  const overall = Math.round(
    COMPARISON_DOMAINS.reduce((sum, domain) => sum + domains[domain.key], 0) /
      COMPARISON_DOMAINS.length
  );
  const topics = parseCsv(topicCsvText).map((row) => ({
    mode: row.mode,
    label: row.label,
    domain: row.domain,
    reviewCount: num(row.review_count),
    signal: num(row.signal),
    x: num(row.x),
    y: num(row.y),
    terms: (row.terms || "").split("|").filter(Boolean),
  }));

  return {
    id,
    label: displayCompanyLabel(id),
    overall,
    reviews: num(companyRow.n_reviews),
    domains,
    topics,
  };
}

export function scoreClass(score: number): string {
  if (score >= 80) return "bg-primary text-primary-foreground";
  if (score >= 70) return "bg-olive-muted/70 text-foreground";
  return "bg-destructive/15 text-destructive";
}
