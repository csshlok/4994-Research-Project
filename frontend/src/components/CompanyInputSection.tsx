import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search, ArrowRight, ChevronDown } from "lucide-react";
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

  const hasTyped = searchText.trim().length > 0;
  const shouldShowDropdown =
    isFocused && showDropdown && hasTyped && filteredCompanies.length > 0;

  return (
    <section className="py-24 px-6 bg-muted/30" id="company-input">
      <div className="max-w-4xl mx-auto text-center">
        <div className="inline-flex items-center rounded-full border border-border bg-background px-4 py-2 mb-6">
          <span className="text-sm font-medium text-muted-foreground">
            Start Analysis
          </span>
        </div>

        <h2 className="text-4xl md:text-5xl font-display font-semibold text-foreground mb-6">
          Analyze a Company
        </h2>

        <p className="text-xl text-muted-foreground mb-12 max-w-2xl mx-auto leading-relaxed">
          Start typing a company name to search from the scraped company list.
        </p>

        <form onSubmit={handleSubmit} className="max-w-2xl mx-auto">
          <div
            className={`relative flex flex-col rounded-2xl border bg-background shadow-lg transition-all duration-300 ${
              isFocused
                ? "border-primary shadow-xl shadow-primary/10"
                : "border-border hover:border-primary/50"
            }`}
          >
            <div className="flex items-center px-6 py-4">
              <Search className="w-5 h-5 text-muted-foreground mr-4 flex-shrink-0" />

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
                placeholder="Type a company name..."
                className="flex-1 border-0 bg-transparent text-lg placeholder:text-muted-foreground/60 focus-visible:ring-0 focus-visible:ring-offset-0 h-12"
                maxLength={100}
              />

              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={() => {
                  setIsFocused(true);
                  setShowDropdown(true);
                }}
                className="ml-2 p-2 rounded-lg hover:bg-muted transition-colors"
                aria-label="Search companies"
              >
                <ChevronDown className="w-5 h-5 text-muted-foreground" />
              </button>

              <Button
                type="submit"
                size="lg"
                className="ml-4 px-8 rounded-xl font-medium"
                disabled={!searchText.trim()}
              >
                Run Analysis
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </div>

            {shouldShowDropdown && (
              <div className="border-t border-border px-4 py-3 max-h-80 overflow-y-auto text-left">
                <div className="grid gap-2">
                  {filteredCompanies.map((company) => (
                    <button
                      key={company.value}
                      type="button"
                      onMouseDown={(e) => e.preventDefault()}
                      onClick={() => handleSelect(company.label)}
                      className="w-full rounded-xl border border-transparent px-4 py-3 text-left transition-colors hover:bg-muted hover:border-border"
                    >
                      <div className="font-medium text-foreground">
                        {company.label}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {errorMessage && (
            <p className="text-sm text-red-500 mt-4 text-left">
              {errorMessage}
            </p>
          )}

          {!errorMessage && hasTyped && !exactMatch && (
            <p className="text-sm text-amber-600 mt-4 text-left">
              No exact company match found yet. Keep typing or select one from
              the suggestions.
            </p>
          )}

          {!errorMessage && (
            <p className="text-sm text-muted-foreground mt-4 text-left">
              Suggestions appear only for companies starting with the typed characters.
            </p>
          )}
        </form>
      </div>
    </section>
  );
}