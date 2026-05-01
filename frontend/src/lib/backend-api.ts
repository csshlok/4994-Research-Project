const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  import.meta.env.VITE_API_BASE ||
  "http://localhost:8000"
).replace(/\/+$/, "");

export interface RunResponse {
  job_id: string;
}

export interface JobStage {
  current?: string | null;
  status?: string | null;
}

export interface JobStatusResponse {
  status: string;
  message?: string;
  company_id_requested?: string;
  company_id_resolved?: string;
  run_dir?: string;
  bundle_path?: string;
  rc?: number | null;
  stage?: JobStage;
}

export interface JobOutputsResponse {
  files: string[];
}

export interface ScoredCompany {
  id: string;
  has_review_scores: boolean;
  has_company_scores: boolean;
  has_cleaned_reviews: boolean;
  has_topics: boolean;
  has_rag?: boolean;
}

export interface ScoredCompaniesResponse {
  companies: ScoredCompany[];
}

export interface ScoredCompanyOutputsResponse {
  company_id: string;
  files: string[];
}

export interface ScoredCompanyRagResponse {
  company_id: string;
  summary?: unknown;
  clusters?: unknown;
  insights?: unknown;
  evidence?: unknown;
  profile?: unknown;
}

export interface ComparisonRagSummary {
  schema_version?: number;
  source?: string;
  model?: string;
  generated_at?: string;
  executive_summary: string[];
  key_differences: string[];
  shared_risks: string[];
  company_notes: Array<{
    company: string;
    strength: string;
    risk: string;
  }>;
  best_fit_by_need: Array<{
    need: string;
    company: string;
    reason: string;
  }>;
}

function apiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${clean}`;
}

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), init);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status} for ${path}: ${txt || res.statusText}`);
  }
  return (await res.json()) as T;
}

async function fetchText(path: string, init?: RequestInit): Promise<string> {
  const res = await fetch(apiUrl(path), init);
  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`HTTP ${res.status} for ${path}: ${txt || res.statusText}`);
  }
  return await res.text();
}

export async function startPipelineJob(companyName: string): Promise<RunResponse> {
  return await fetchJson<RunResponse>("/api/run", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      mode: "cache",
      company_name: companyName,
    }),
  });
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return await fetchJson<JobStatusResponse>(`/api/job/${encodeURIComponent(jobId)}`);
}

export async function getJobOutputs(jobId: string): Promise<JobOutputsResponse> {
  return await fetchJson<JobOutputsResponse>(`/api/job/${encodeURIComponent(jobId)}/outputs`);
}

export function getJobDownloadUrl(jobId: string, path: string): string {
  return apiUrl(
    `/api/job/${encodeURIComponent(jobId)}/download?path=${encodeURIComponent(path)}`
  );
}

export async function downloadJobFileText(jobId: string, path: string): Promise<string> {
  const url = `/api/job/${encodeURIComponent(jobId)}/download?path=${encodeURIComponent(path)}`;
  return await fetchText(url);
}

export async function getScoredCompanies(): Promise<ScoredCompaniesResponse> {
  return await fetchJson<ScoredCompaniesResponse>("/api/scored-companies");
}

export async function getScoredCompanyOutputs(companyId: string): Promise<ScoredCompanyOutputsResponse> {
  return await fetchJson<ScoredCompanyOutputsResponse>(
    `/api/scored-company/${encodeURIComponent(companyId)}/outputs`
  );
}

export async function getScoredCompanyRag(companyId: string): Promise<ScoredCompanyRagResponse> {
  return await fetchJson<ScoredCompanyRagResponse>(
    `/api/scored-company/${encodeURIComponent(companyId)}/rag`
  );
}

export async function generateComparisonRagSummary(companies: string[]): Promise<ComparisonRagSummary> {
  return await fetchJson<ComparisonRagSummary>("/api/compare/rag-summary", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ companies }),
  });
}

export function getScoredCompanyDownloadUrl(companyId: string, path: string): string {
  return apiUrl(
    `/api/scored-company/${encodeURIComponent(companyId)}/download?path=${encodeURIComponent(path)}`
  );
}

export async function downloadScoredCompanyFileText(companyId: string, path: string): Promise<string> {
  const url = `/api/scored-company/${encodeURIComponent(companyId)}/download?path=${encodeURIComponent(path)}`;
  return await fetchText(url);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}
