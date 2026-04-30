export function SummarySection() {
  return (
    <section id="summary" className="py-32 px-6 bg-primary/5">
      <div className="container-wide max-w-4xl mx-auto text-center">

        <span className="inline-block text-sm font-medium text-primary uppercase tracking-wider mb-4">
          Overview
        </span>

        <h2 className="font-serif text-3xl md:text-4xl font-semibold text-foreground mb-6">
          Summary
        </h2>

        <p className="text-muted-foreground text-lg leading-relaxed mb-6">
          This project analyzes employee reviews to understand how people describe their
          experience at work. It looks beyond a simple positive or negative rating by
          connecting review language to five human needs: basic comfort and stability,
          safety and fairness, belonging, recognition and growth, and work-life support.
          The goal is to make large sets of reviews easier to interpret without requiring
          readers to manually sort through hundreds of individual comments.
        </p>

        <p className="text-muted-foreground text-lg leading-relaxed">
          The pipeline cleans the review text, identifies meaningful language patterns,
          scores overall sentiment, and measures whether each need is being fulfilled or
          hindered. The final results summarize where a company appears strongest, where
          employees describe recurring friction, and which themes deserve closer attention.
          These scores are not absolute judgments; they are structured signals that help
          explain what employees are saying in a clear, evidence-based way.
        </p>

      </div>
    </section>
  );
}
