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
      " Provides for basic needs such as food and shelter, and supports physical and mental health.",
    examples:
      "good pay, healthcare, stability; hinders--burnout, unhealthy, fatigue",
    icon: <Heart className="w-6 h-6" />,
  },
  {
    number: 2,
    title: "Self-Protection",
    description:
      " Feeling safe from people who might hurt or exploit me.",
    examples:
    "thical work culture, empathy, boundaries honored; hinders-- toxic, fear, harassment",
    icon: <Shield className="w-6 h-6" />,
  },
  {
    number: 3,
    title: "Affiliation",
    description:
    "Fosters social connection with others and helps to make me desirable to potential friends, allies and romantic partners.",
    examples:
      " belonging, family-like, inclusive, social; hinders-- isolation, disconnected, unfriendly",
    icon: <Users className="w-6 h-6" />,
  },
  {
    number: 4,
    title: "Status & Esteem",
    description:
    "Feeling respected and valued for my skills and contributions.",
    examples:
     "prestige, leadership opportunities, education expenses; hinders-- disrespected, undervalued, no growth",
    icon: <Award className="w-6 h-6" />,
  },
  {
    number: 5,
    title: "Family Care",
    description:
    "Facilitates connecting with and supporting my family, including promoting a good relationship with my committed partner.",
    examples:
   "family-friendly, personal life respected, maternal/maternity leave; hinders-- unpredictable hours, no work-life balance, inflexible schedule",
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
