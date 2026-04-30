import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, ArrowRight, GitCompare, Check, Plus } from "lucide-react";
import { COMPANY_OPTIONS } from "@/lib/company-options";

interface CompanyInputSectionProps {
  onSubmit: (companyName: string) => void;
  onCompareSubmit: (companyNames: string[]) => void;
  initialCompareCompanies?: string[];
}

export function CompanyInputSection({
  onSubmit,
  onCompareSubmit,
  initialCompareCompanies = [],
}: CompanyInputSectionProps) {
  const [searchText, setSearchText] = useState("");
  const [compareSearchText, setCompareSearchText] = useState("");
  const [mode, setMode] = useState<"single" | "compare">(
    initialCompareCompanies.length > 0 ? "compare" : "single"
  );
  const [selectedCompare, setSelectedCompare] = useState<string[]>(initialCompareCompanies);
  const [comparePage, setComparePage] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const sortedCompanies = useMemo(() => {
    return [...COMPANY_OPTIONS].sort((a, b) =>
      a.label.localeCompare(b.label)
    );
  }, []);

  const filteredCompanies = useMemo(() => {
    const query = searchText.trim().toLowerCase();

    if (!query) {
      return [];
    }

    return sortedCompanies.filter((company) =>
      company.label.toLowerCase().startsWith(query)
    );
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
    if (compareSearchText.trim()) {
      return 1;
    }
    return Math.max(1, Math.ceil(sortedCompanies.length / 13));
  }, [compareSearchText, sortedCompanies.length]);

  const exactMatch = useMemo(() => {
    const query = searchText.trim().toLowerCase();

    return sortedCompanies.find(
      (company) => company.label.toLowerCase() === query
    );
  }, [searchText, sortedCompanies]);

  const handleSelect = (companyLabel: string) => {
    setSearchText(companyLabel);
    setErrorMessage("");
    setShowDropdown(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const selectedCompany = sortedCompanies.find(
      (company) =>
        company.label.toLowerCase() === searchText.trim().toLowerCase()
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

  const hasTyped = searchText.trim().length > 0;

  const shouldShowDropdown =
    isFocused &&
    showDropdown &&
    hasTyped;

  return (
    <section className="py-16 px-6 bg-background" id="company-input">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-3xl border border-border/30 shadow-sm px-6 md:px-8 py-8 md:py-10">
          <div className="max-w-2xl mx-auto text-center">
            <p className="text-xs font-medium tracking-[0.18em] text-primary uppercase mb-4">
              Start Analysis
            </p>

            <h2 className="text-2xl md:text-4xl font-display font-semibold text-foreground mb-4">
              {mode === "single" ? "Analyze a Company" : "Compare Companies"}
            </h2>

            <p className="text-base text-muted-foreground mb-8 max-w-xl mx-auto leading-relaxed">
              {mode === "single"
                ? "Enter a company name to receive a comprehensive analysis of employee experiences mapped to five human needs."
                : "Select multiple companies to compare their scores, goal domains, and employee language clusters."}
            </p>

            <div className="mb-6 inline-flex rounded-xl border border-border bg-background p-1">
              <button
                type="button"
                onClick={() => {
                  setMode("single");
                  setErrorMessage("");
                }}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  mode === "single"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Analyze One
              </button>
              <button
                type="button"
                onClick={() => {
                    setMode("compare");
                    setComparePage(0);
                    setErrorMessage("");
                }}
                className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  mode === "compare"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Compare Companies
              </button>
            </div>

            {mode === "single" ? (
            <form onSubmit={handleSubmit} className="max-w-xl mx-auto">
              <div
                className={`relative flex flex-col rounded-xl border bg-background shadow-sm transition-all duration-300 ${
                  isFocused ? "border-primary/40" : "border-border/60"
                }`}
              >
                <div className="flex items-center gap-2 px-3 py-2">
                  <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-muted shrink-0">
                    <Search className="w-4 h-4 text-primary/80" />
                  </div>

                  <Input
                    value={searchText}
                    onChange={(e) => {
                      setSearchText(e.target.value);
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
                    className="flex-1 border-0 bg-transparent text-sm placeholder:text-muted-foreground/70 focus-visible:ring-0 focus-visible:ring-offset-0 h-9 px-0"
                    maxLength={100}
                  />

                  <Button
                    type="submit"
                    size="sm"
                    className="px-5 rounded-lg font-semibold shrink-0"
                    disabled={!searchText.trim()}
                  >
                    Run Analysis
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                </div>

                {shouldShowDropdown && (
                  <div className="border-t border-border px-3 py-2 max-h-60 overflow-y-auto text-left bg-background rounded-b-xl">
                    <div className="grid gap-1">
                      {filteredCompanies.length > 0 ? (
                        filteredCompanies.map((company) => (
                          <button
                            key={company.value}
                            type="button"
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => handleSelect(company.label)}
                            className="w-full rounded-lg border border-transparent px-3 py-2 text-left text-sm transition-colors hover:bg-muted hover:border-border"
                          >
                            {company.label}
                          </button>
                        ))
                      ) : (
                        <div className="px-3 py-3 text-sm text-muted-foreground text-center">
                          Don’t see your company? We haven’t added it to our review database.
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {errorMessage && (
                <p className="text-sm text-red-500 mt-3 text-left">
                  {errorMessage}
                </p>
              )}

              {!errorMessage && hasTyped && !exactMatch && filteredCompanies.length > 0 && (
                <p className="text-sm text-amber-600 mt-3 text-left">
                  No exact company match found yet. Keep typing or select one from
                  the suggestions.
                </p>
              )}

              {!errorMessage && (
                <p className="text-sm text-muted-foreground mt-4">
                  Cached analyses load in a few seconds.
                </p>
              )}
            </form>
            ) : (
              <div className="max-w-3xl mx-auto text-left">
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
                {errorMessage && (
                  <p className="mb-3 text-sm text-red-500">{errorMessage}</p>
                )}
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
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
