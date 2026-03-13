import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, ArrowRight } from "lucide-react";
import { COMPANY_OPTIONS } from "@/lib/company-options";

interface CompanyInputSectionProps {
  onSubmit: (companyName: string) => void;
}

export function CompanyInputSection({
  onSubmit,
}: CompanyInputSectionProps) {
  const [searchText, setSearchText] = useState("");
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

  const hasTyped = searchText.trim().length > 0;

  const shouldShowDropdown =
    isFocused &&
    showDropdown &&
    hasTyped &&
    filteredCompanies.length > 0;

  return (
    <section className="py-16 px-6 bg-background" id="company-input">
      <div className="max-w-4xl mx-auto">

        {/* Smaller white section container */}
        <div className="bg-white rounded-3xl border border-border/30 shadow-sm px-6 md:px-8 py-8 md:py-10">

          <div className="max-w-2xl mx-auto text-center">

            {/* START ANALYSIS */}
            <p className="text-xs font-medium tracking-[0.18em] text-primary uppercase mb-4">
              Start Analysis
            </p>

            <h2 className="text-2xl md:text-4xl font-display font-semibold text-foreground mb-4">
              Analyze a Company
            </h2>

            <p className="text-base text-muted-foreground mb-8 max-w-xl mx-auto leading-relaxed">
              Enter a company name to receive a comprehensive analysis of employee
              experiences mapped to the seven social domains.
            </p>

            <form onSubmit={handleSubmit} className="max-w-xl mx-auto">

              <div
                className={`relative flex flex-col rounded-xl border bg-background shadow-sm transition-all duration-300 ${
                  isFocused ? "border-primary/40" : "border-border/60"
                }`}
              >

                <div className="flex items-center gap-2 px-3 py-2">

                  {/* Search icon */}
                  <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-muted shrink-0">
                    <Search className="w-4 h-4 text-primary/80" />
                  </div>

                  {/* Input */}
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

                  {/* Run button */}
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

                      {filteredCompanies.map((company) => (
                        <button
                          key={company.value}
                          type="button"
                          onMouseDown={(e) => e.preventDefault()}
                          onClick={() => handleSelect(company.label)}
                          className="w-full rounded-lg border border-transparent px-3 py-2 text-left text-sm transition-colors hover:bg-muted hover:border-border"
                        >
                          {company.label}
                        </button>
                      ))}

                    </div>
                  </div>
                )}

              </div>

              {errorMessage && (
                <p className="text-sm text-red-500 mt-3 text-left">
                  {errorMessage}
                </p>
              )}

              {!errorMessage && hasTyped && !exactMatch && (
                <p className="text-sm text-amber-600 mt-3 text-left">
                  No exact company match found yet. Keep typing or select one from
                  the suggestions.
                </p>
              )}

              {!errorMessage && (
                <p className="text-sm text-muted-foreground mt-4">
                  Analysis typically takes 2-3 minutes depending on review volume
                </p>
              )}

            </form>

          </div>

        </div>

      </div>
    </section>
  );
}