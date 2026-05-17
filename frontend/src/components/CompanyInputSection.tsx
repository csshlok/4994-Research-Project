import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Search, ArrowRight, GitCompare, Check, Plus, Sparkles } from "lucide-react";
import { COMPANY_OPTIONS } from "@/lib/company-options";
import { MatchTradeoffControls } from "@/components/MatchTradeoffControls";
import { type MatchAcceptanceKey, type MatchInput, type MatchTradeoffKey, validateMatchInput } from "@/lib/matching";

interface CompanyInputSectionProps {
  onSubmit: (companyName: string) => void;
  onCompareSubmit: (companyNames: string[]) => void;
  onMatchSubmit: (input: MatchInput) => void;
  initialCompareCompanies?: string[];
}

type InputMode = "single" | "compare" | "match";

export function CompanyInputSection({
  onSubmit,
  onCompareSubmit,
  onMatchSubmit,
  initialCompareCompanies = [],
}: CompanyInputSectionProps) {
  const [searchText, setSearchText] = useState("");
  const [compareSearchText, setCompareSearchText] = useState("");
  const [matchText, setMatchText] = useState("");
  const [matchTradeoffs, setMatchTradeoffs] = useState<MatchTradeoffKey[]>([]);
  const [acceptedTradeoffs, setAcceptedTradeoffs] = useState<MatchAcceptanceKey[]>([]);
  const [mode, setMode] = useState<InputMode>(
    initialCompareCompanies.length > 0 ? "compare" : "single"
  );
  const [selectedCompare, setSelectedCompare] = useState<string[]>(initialCompareCompanies);
  const [comparePage, setComparePage] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const sortedCompanies = useMemo(() => {
    return [...COMPANY_OPTIONS].sort((a, b) => a.label.localeCompare(b.label));
  }, []);

  const filteredCompanies = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    if (!query) return [];
    return sortedCompanies.filter((company) => company.label.toLowerCase().startsWith(query));
  }, [searchText, sortedCompanies]);

  const filteredCompareCompanies = useMemo(() => {
    const query = compareSearchText.trim().toLowerCase();
    const base = query
      ? sortedCompanies.filter((company) => company.label.toLowerCase().includes(query))
      : sortedCompanies;
    const start = query ? 0 : comparePage * 13;
    return base.slice(start, start + 13);
  }, [compareSearchText, sortedCompanies, comparePage]);

  const compareTotalPages = useMemo(() => {
    if (compareSearchText.trim()) return 1;
    return Math.max(1, Math.ceil(sortedCompanies.length / 13));
  }, [compareSearchText, sortedCompanies.length]);

  const exactMatch = useMemo(() => {
    const query = searchText.trim().toLowerCase();
    return sortedCompanies.find((company) => company.label.toLowerCase() === query);
  }, [searchText, sortedCompanies]);

  const handleSelect = (companyLabel: string) => {
    setSearchText(companyLabel);
    setErrorMessage("");
    setShowDropdown(false);
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    const selectedCompany = sortedCompanies.find(
      (company) => company.label.toLowerCase() === searchText.trim().toLowerCase()
    );

    if (!selectedCompany) {
      setErrorMessage(
        "This company has not been added to the scraped review dataset yet. Please select one of the available companies from the list."
      );
      return;
    }

    setErrorMessage("");
    onSubmit(selectedCompany.value);
  };

  const toggleCompareCompany = (companyValue: string) => {
    setSelectedCompare((cur) => {
      if (cur.includes(companyValue)) {
        return cur.filter((value) => value !== companyValue);
      }
      if (cur.length >= 3) {
        return cur;
      }
      return [...cur, companyValue];
    });
  };

  const handleCompareSubmit = () => {
    if (selectedCompare.length < 2) {
      setErrorMessage("Select at least two companies to compare.");
      return;
    }
    setErrorMessage("");
    onCompareSubmit(selectedCompare);
  };

  const handleMatchSubmit = () => {
    const input = {
      narrative: matchText,
      tradeoffs: matchTradeoffs,
      acceptedTradeoffs,
    };
    const validationError = validateMatchInput(input);
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }
    setErrorMessage("");
    onMatchSubmit(input);
  };

  const hasTyped = searchText.trim().length > 0;
  const shouldShowDropdown = isFocused && showDropdown && hasTyped;

  return (
    <section className="bg-background px-6 py-16" id="company-input">
      <div className="mx-auto max-w-4xl">
        <div className="rounded-3xl border border-border/30 bg-white px-6 py-8 shadow-sm md:px-8 md:py-10">
          <div className="mx-auto max-w-2xl text-center">
            <p className="mb-4 text-xs font-medium uppercase tracking-[0.18em] text-primary">
              Start Analysis
            </p>

            <h2 className="mb-4 text-2xl font-display font-semibold text-foreground md:text-4xl">
              {mode === "single"
                ? "Analyze a Company"
                : mode === "compare"
                ? "Compare Companies"
                : "Match Yourself"}
            </h2>

            <p className="mx-auto mb-8 max-w-xl text-base leading-relaxed text-muted-foreground">
              {mode === "single"
                ? "Enter a company name to receive a comprehensive analysis of employee experiences mapped to five human needs."
                : mode === "compare"
                ? "Select multiple companies to compare their scores, goal domains, and employee language clusters."
                : "Describe what matters to you at work, then choose the tradeoffs you care about most."}
            </p>

            <div className="mb-6 inline-flex flex-wrap justify-center rounded-xl border border-border bg-background p-1">
              {[
                { key: "single", label: "Analyze One" },
                { key: "compare", label: "Compare Companies" },
                { key: "match", label: "Match Yourself" },
              ].map((item) => (
                <button
                  key={item.key}
                  type="button"
                  onClick={() => {
                    setMode(item.key as InputMode);
                    setErrorMessage("");
                    if (item.key === "compare") setComparePage(0);
                  }}
                  className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                    mode === item.key
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {mode === "single" ? (
              <form onSubmit={handleSubmit} className="mx-auto max-w-xl">
                <div
                  className={`relative flex flex-col rounded-xl border bg-background shadow-sm transition-all duration-300 ${
                    isFocused ? "border-primary/40" : "border-border/60"
                  }`}
                >
                  <div className="flex items-center gap-2 px-3 py-2">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted">
                      <Search className="h-4 w-4 text-primary/80" />
                    </div>

                    <Input
                      value={searchText}
                      onChange={(event) => {
                        setSearchText(event.target.value);
                        setErrorMessage("");
                        setShowDropdown(true);
                      }}
                      onFocus={() => {
                        setIsFocused(true);
                        setShowDropdown(true);
                      }}
                      onBlur={() => {
                        window.setTimeout(() => {
                          setIsFocused(false);
                          setShowDropdown(false);
                        }, 150);
                      }}
                      placeholder="Enter company name..."
                      className="h-9 flex-1 border-0 bg-transparent px-0 text-sm placeholder:text-muted-foreground/70 focus-visible:ring-0 focus-visible:ring-offset-0"
                      maxLength={100}
                    />

                    <Button
                      type="submit"
                      size="sm"
                      className="shrink-0 rounded-lg px-5 font-semibold"
                      disabled={!searchText.trim()}
                    >
                      Run Analysis
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Button>
                  </div>

                  {shouldShowDropdown && (
                    <div className="max-h-60 overflow-y-auto rounded-b-xl border-t border-border bg-background px-3 py-2 text-left">
                      <div className="grid gap-1">
                        {filteredCompanies.length > 0 ? (
                          filteredCompanies.map((company) => (
                            <button
                              key={company.value}
                              type="button"
                              onMouseDown={(event) => event.preventDefault()}
                              onClick={() => handleSelect(company.label)}
                              className="w-full rounded-lg border border-transparent px-3 py-2 text-left text-sm transition-colors hover:border-border hover:bg-muted"
                            >
                              {company.label}
                            </button>
                          ))
                        ) : (
                          <div className="px-3 py-3 text-center text-sm text-muted-foreground">
                            Don't see your company? We haven't added it to our review database.
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>

                {errorMessage && <p className="mt-3 text-left text-sm text-red-500">{errorMessage}</p>}

                {!errorMessage && hasTyped && !exactMatch && filteredCompanies.length > 0 && (
                  <p className="mt-3 text-left text-sm text-amber-600">
                    No exact company match found yet. Keep typing or select one from the suggestions.
                  </p>
                )}

                {!errorMessage && (
                  <p className="mt-4 text-sm text-muted-foreground">Cached analyses load in a few seconds.</p>
                )}
              </form>
            ) : mode === "compare" ? (
              <div className="mx-auto max-w-3xl text-left">
                <div className="mb-4 flex items-center gap-2 rounded-xl border border-border bg-background px-3 py-2">
                  <Search className="h-4 w-4 text-primary" />
                  <Input
                    value={compareSearchText}
                    onChange={(event) => {
                      setCompareSearchText(event.target.value);
                      setComparePage(0);
                    }}
                    placeholder="Search companies to compare..."
                    className="border-0 bg-transparent text-sm focus-visible:ring-0 focus-visible:ring-offset-0"
                  />
                </div>
                <div className="mb-5 flex flex-wrap gap-2">
                  {filteredCompareCompanies.map((company) => {
                    const selected = selectedCompare.includes(company.value);
                    return (
                      <button
                        key={company.value}
                        type="button"
                        onClick={() => toggleCompareCompany(company.value)}
                        className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-sm transition-colors ${
                          selected
                            ? "border-primary bg-primary text-primary-foreground"
                            : "border-border bg-card hover:border-primary/40"
                        }`}
                      >
                        {selected ? <Check className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
                        {company.label}
                      </button>
                    );
                  })}
                  {!compareSearchText.trim() && compareTotalPages > 1 && (
                    <button
                      type="button"
                      onClick={() => setComparePage((page) => (page + 1) % compareTotalPages)}
                      className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-background px-3 py-2 text-sm font-medium text-primary transition-colors hover:bg-primary hover:text-primary-foreground"
                    >
                      Next Set
                      <ArrowRight className="h-4 w-4" />
                    </button>
                  )}
                </div>
                {errorMessage && <p className="mb-3 text-sm text-red-500">{errorMessage}</p>}
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <p className="text-sm text-muted-foreground">
                    {selectedCompare.length} selected. Choose 2-3 companies.
                  </p>
                  <Button onClick={handleCompareSubmit} disabled={selectedCompare.length < 2} className="gap-2">
                    <GitCompare className="h-4 w-4" />
                    Run Comparison
                  </Button>
                </div>
              </div>
            ) : (
              <div className="mx-auto max-w-3xl text-left">
                <Textarea
                  value={matchText}
                  onChange={(event) => {
                    setMatchText(event.target.value);
                    setErrorMessage("");
                  }}
                  className="min-h-[150px] resize-none bg-background text-base leading-relaxed"
                  placeholder="Example: I want fair managers, flexibility, strong growth, and a culture that does not reward burnout."
                />

                <div className="mt-5">
                  <MatchTradeoffControls
                    selected={matchTradeoffs}
                    accepted={acceptedTradeoffs}
                    onChange={(next) => {
                      setMatchTradeoffs(next);
                      setErrorMessage("");
                    }}
                    onAcceptedChange={(next) => {
                      setAcceptedTradeoffs(next);
                      setErrorMessage("");
                    }}
                  />
                </div>

                {errorMessage && <p className="mt-3 text-sm text-red-500">{errorMessage}</p>}

                <div className="mt-6 flex justify-end">
                  <Button onClick={handleMatchSubmit} className="gap-2">
                    <Sparkles className="h-4 w-4" />
                    Find My Matches
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
