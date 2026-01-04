import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Download,
  FileSpreadsheet,
  BarChart3,
  ArrowLeft,
  Calendar,
  FileText,
  TrendingUp,
  TrendingDown,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

interface ResultsPageProps {
  companyName: string;
  onBack: () => void;
}

// Mock data for demonstration
const domainScores = [
  { domain: "Physiological", fulfillment: 72, hindrance: 28, short: "Phys" },
  { domain: "Self-Protection", fulfillment: 65, hindrance: 35, short: "Prot" },
  { domain: "Affiliation", fulfillment: 81, hindrance: 19, short: "Affil" },
  { domain: "Status & Esteem", fulfillment: 58, hindrance: 42, short: "Status" },
  { domain: "Family Care", fulfillment: 45, hindrance: 55, short: "Family" },
  { domain: "Mate Acquisition", fulfillment: 68, hindrance: 32, short: "Acq" },
  { domain: "Mate Retention", fulfillment: 62, hindrance: 38, short: "Ret" },
];

const radarData = domainScores.map((d) => ({
  domain: d.short,
  score: d.fulfillment,
  fullMark: 100,
}));

export function ResultsPage({ companyName, onBack }: ResultsPageProps) {
  const reviewCount = Math.floor(Math.random() * 5000) + 500;
  const analysisDate = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  const overallScore = Math.round(
    domainScores.reduce((acc, d) => acc + d.fulfillment, 0) / domainScores.length
  );

  const strongestDomain = domainScores.reduce((a, b) =>
    a.fulfillment > b.fulfillment ? a : b
  );
  const weakestDomain = domainScores.reduce((a, b) =>
    a.fulfillment < b.fulfillment ? a : b
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container-wide py-4 flex items-center justify-between">
          <Button variant="ghost" onClick={onBack} className="gap-2">
            <ArrowLeft className="w-4 h-4" />
            New Analysis
          </Button>
          <div className="text-sm text-muted-foreground flex items-center gap-2">
            <Calendar className="w-4 h-4" />
            {analysisDate}
          </div>
        </div>
      </header>

      <main className="container-wide section-padding">
        {/* Overview */}
        <section className="mb-16">
          <div className="text-center mb-8">
            <span className="inline-block text-sm font-medium text-primary uppercase tracking-wider mb-2">
              Analysis Complete
            </span>
            <h1 className="font-serif text-4xl md:text-5xl font-semibold text-foreground mb-4">
              {companyName}
            </h1>
            <p className="text-muted-foreground">
              Based on {reviewCount.toLocaleString()} employee reviews
            </p>
          </div>

          {/* Summary cards */}
          <div className="grid sm:grid-cols-3 gap-4 max-w-3xl mx-auto">
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="text-4xl font-serif font-bold text-primary mb-2">
                  {overallScore}%
                </div>
                <p className="text-sm text-muted-foreground">Overall Score</p>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <TrendingUp className="w-5 h-5 text-primary" />
                  <span className="text-lg font-medium">{strongestDomain.domain}</span>
                </div>
                <p className="text-sm text-muted-foreground">Strongest Domain</p>
              </CardContent>
            </Card>
            <Card className="text-center">
              <CardContent className="pt-6">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <TrendingDown className="w-5 h-5 text-destructive" />
                  <span className="text-lg font-medium">{weakestDomain.domain}</span>
                </div>
                <p className="text-sm text-muted-foreground">Needs Attention</p>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Summary paragraph */}
        <section className="mb-16">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                Executive Summary
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed">
                {companyName} demonstrates strong performance in the <strong className="text-foreground">{strongestDomain.domain}</strong> domain 
                ({strongestDomain.fulfillment}% positive sentiment), indicating employees feel particularly supported in this area. 
                However, the analysis reveals opportunities for improvement in <strong className="text-foreground">{weakestDomain.domain}</strong> 
                ({weakestDomain.fulfillment}% positive), where employee experiences suggest unmet needs. 
                Overall, the company achieves an aggregate score of {overallScore}% across all seven social domains, 
                reflecting a {overallScore >= 70 ? "generally positive" : overallScore >= 50 ? "mixed" : "challenging"} employee experience landscape.
              </p>
            </CardContent>
          </Card>
        </section>

        {/* Downloads */}
        <section className="mb-16">
          <h2 className="font-serif text-2xl font-semibold text-foreground mb-6">
            Download Data
          </h2>
          <div className="grid sm:grid-cols-3 gap-4">
            <Card className="group hover:shadow-elevated transition-shadow cursor-pointer">
              <CardContent className="pt-6 text-center">
                <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center mx-auto mb-4 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  <FileSpreadsheet className="w-6 h-6" />
                </div>
                <h3 className="font-medium mb-1">Cleaned Reviews</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Pre-processed review data
                </p>
                <Button variant="outline" size="sm" className="gap-2">
                  <Download className="w-4 h-4" />
                  CSV
                </Button>
              </CardContent>
            </Card>
            <Card className="group hover:shadow-elevated transition-shadow cursor-pointer">
              <CardContent className="pt-6 text-center">
                <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center mx-auto mb-4 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  <BarChart3 className="w-6 h-6" />
                </div>
                <h3 className="font-medium mb-1">Review Scores</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Goal scores per review
                </p>
                <Button variant="outline" size="sm" className="gap-2">
                  <Download className="w-4 h-4" />
                  CSV
                </Button>
              </CardContent>
            </Card>
            <Card className="group hover:shadow-elevated transition-shadow cursor-pointer">
              <CardContent className="pt-6 text-center">
                <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center mx-auto mb-4 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                  <FileText className="w-6 h-6" />
                </div>
                <h3 className="font-medium mb-1">Aggregated Scores</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Company-level summary
                </p>
                <Button variant="outline" size="sm" className="gap-2">
                  <Download className="w-4 h-4" />
                  CSV
                </Button>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Visualizations */}
        <section className="mb-16">
          <h2 className="font-serif text-2xl font-semibold text-foreground mb-6">
            Visualizations
          </h2>
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Bar chart */}
            <Card>
              <CardHeader>
                <CardTitle>Goal Fulfillment vs. Hindrance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={domainScores}
                      layout="vertical"
                      margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis type="number" domain={[0, 100]} stroke="hsl(var(--muted-foreground))" />
                      <YAxis
                        dataKey="domain"
                        type="category"
                        stroke="hsl(var(--muted-foreground))"
                        tick={{ fontSize: 12 }}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                      <Bar dataKey="fulfillment" fill="hsl(var(--primary))" name="Fulfillment %" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Radar chart */}
            <Card>
              <CardHeader>
                <CardTitle>Domain Profile</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                      <PolarGrid stroke="hsl(var(--border))" />
                      <PolarAngleAxis
                        dataKey="domain"
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 12 }}
                      />
                      <PolarRadiusAxis
                        angle={90}
                        domain={[0, 100]}
                        tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 10 }}
                      />
                      <Radar
                        name="Score"
                        dataKey="score"
                        stroke="hsl(var(--primary))"
                        fill="hsl(var(--primary))"
                        fillOpacity={0.3}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--card))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: "8px",
                        }}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </section>

        {/* Interpretation */}
        <section>
          <Card className="bg-muted/30">
            <CardHeader>
              <CardTitle>How to Read These Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-muted-foreground">
              <p>
                <strong className="text-foreground">Fulfillment scores</strong> indicate the percentage of review content 
                expressing positive experiences related to each social domain. Higher scores suggest the company 
                effectively supports that fundamental need.
              </p>
              <p>
                <strong className="text-foreground">Important reminder:</strong> These results reflect aggregated 
                employee experiences as expressed in public reviews. They represent patterns in how employees 
                communicate about their workplace, not objective ground truth about company policies or practices.
              </p>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}
