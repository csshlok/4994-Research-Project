import { DomainCard } from "./DomainCard";
import {
  Heart,
  Shield,
  Users,
  Award,
  Home,
  Sparkles,
  Anchor,
} from "lucide-react";

const domains = [
  {
    number: 1,
    title: "Physiological Needs",
    description:
      "Concerns whether a job supports basic survival and comfort. This domain captures the foundational requirements for employees to sustain themselves.",
    examples:
      "Pay adequacy, workload and exhaustion levels, health benefits, ability to meet basic living needs, physical working conditions.",
    icon: <Heart className="w-6 h-6" />,
  },
  {
    number: 2,
    title: "Self-Protection",
    description:
      "Relates to safety, fairness, and stability at work. This domain reflects whether employees feel secure and treated justly.",
    examples:
      "Psychological safety, freedom from harassment, ethical leadership, job security, predictable policies, fair treatment.",
    icon: <Shield className="w-6 h-6" />,
  },
  {
    number: 3,
    title: "Affiliation",
    description:
      "Captures social belonging at work. This domain examines the quality of interpersonal connections and sense of community.",
    examples:
      "Teamwork quality, inclusion and diversity, workplace friendships, feeling valued by peers, social support vs. isolation.",
    icon: <Users className="w-6 h-6" />,
  },
  {
    number: 4,
    title: "Status & Esteem",
    description:
      "Reflects respect, recognition, and perceived importance within the organization. This domain tracks how employees feel valued.",
    examples:
      "Promotion opportunities, recognition programs, career growth paths, leadership trust, feeling respected vs. overlooked.",
    icon: <Award className="w-6 h-6" />,
  },
  {
    number: 5,
    title: "Family Care",
    description:
      "Concerns how well a job supports life outside of work. This domain measures the organization's respect for personal responsibilities.",
    examples:
      "Work-life balance policies, flexible hours, parental leave, understanding of family responsibilities, remote work options.",
    icon: <Home className="w-6 h-6" />,
  },
];

export function DomainsSection() {
  return (
    <section id="domains" className="section-padding bg-muted/30">
      <div className="container-wide">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <span className="inline-block text-sm font-medium text-primary uppercase tracking-wider mb-4">
            The Framework
          </span>
          <h2 className="font-serif text-3xl md:text-4xl font-semibold text-foreground mb-4">
            Five Social Domains
          </h2>
          <p className="text-muted-foreground text-lg leading-relaxed">
            Our analysis maps employee experiences to fundamental human goals,
            providing a deeper understanding of workplace dynamics beyond surface-level metrics.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-4 max-w-5xl mx-auto">
          {domains.map((domain, index) => (
            <div
              key={domain.number}
              className={index === domains.length - 1 && domains.length % 2 !== 0 ? "md:col-span-2 md:max-w-[calc(50%-0.5rem)] md:mx-auto" : ""}
            >
              <DomainCard
                {...domain}
                delay={index * 100}
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
