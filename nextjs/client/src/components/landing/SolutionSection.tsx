import { motion } from 'framer-motion';
import { SlashDivider } from '@/components/ui/Shapes';
import { Brain, MessageSquare, LayoutDashboard } from 'lucide-react';

const features = [
    {
        icon: <Brain size={40} className="text-[var(--color-brand-primary)]" />,
        title: 'Predictive ML Pipeline',
        subtitle: '7–10 Day Risk Window',
        body: 'Binary classification on telemetry + KPI signals. Walk-forward validation. Anomaly detection as a complementary layer. SHAP explainability for top-5 fault drivers.',
        badge: 'ML MODEL',
        badgeColor: 'var(--color-brand-primary)',
    },
    {
        icon: <MessageSquare size={40} className="text-[var(--color-green-primary)]" />,
        title: 'Generative AI Narratives',
        subtitle: 'RAG-Grounded Insights',
        body: 'Agentic GenAI retrieves inverter data, assesses risk scores, hypothesizes root causes, and auto-drafts maintenance tickets — with hallucination guardrails.',
        badge: 'GEN AI',
        badgeColor: 'var(--color-green-primary)',
    },
    {
        icon: <LayoutDashboard size={40} className="text-[var(--color-warning)]" />,
        title: 'Operator Dashboard',
        subtitle: 'Per-Inverter Risk Intelligence',
        body: 'Real-time risk scores, SHAP feature charts, GenAI narratives, and fault trend visualizations. Built for operators, not data scientists.',
        badge: 'DASHBOARD',
        badgeColor: 'var(--color-warning)',
    },
];

export const SolutionSection = () => {
    return (
        <section className="relative w-full py-24" style={{ backgroundColor: 'var(--color-bg-secondary)' }}>
            <div className="absolute top-0 left-0 w-full" style={{ marginTop: '-80px', zIndex: 11 }}>
                <SlashDivider color="var(--color-bg-secondary)" />
            </div>

            <div className="max-w-6xl mx-auto px-6 relative z-10">
                <div className="text-center mb-16">
                    <motion.h2
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6 }}
                        style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-display)', color: 'var(--color-text-primary)' }}
                    >
                        THE PREDICTION ENGINE
                    </motion.h2>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                        style={{ fontFamily: 'var(--font-body)', fontSize: 'var(--text-h3)', color: 'var(--color-text-secondary)', maxWidth: '600px', margin: '16px auto 0' }}
                    >
                        A multi-layered architecture designed to turn raw telemetry into actionable operational intelligence.
                    </motion.p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {features.map((feature, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 50, filter: 'blur(8px)' }}
                            whileInView={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
                            viewport={{ once: true, amount: 0.3 }}
                            transition={{ duration: 0.65, delay: i * 0.18, ease: [0.25, 0.46, 0.45, 0.94] }}
                            whileHover={{ y: -6, boxShadow: '0 16px 40px #D4900A33' }}
                            style={{
                                background: 'var(--color-bg-primary)',
                                borderRadius: 'var(--radius-lg)',
                                padding: '32px 24px',
                                border: '1px solid var(--color-border)',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '16px'
                            }}
                        >
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-4xl">{feature.icon}</span>
                                <span style={{
                                    fontSize: '0.75rem',
                                    fontWeight: 700,
                                    fontFamily: 'var(--font-sans)',
                                    padding: '4px 10px',
                                    borderRadius: '12px',
                                    backgroundColor: `${feature.badgeColor}22`,
                                    color: feature.badgeColor,
                                    letterSpacing: '0.05em'
                                }}>
                                    {feature.badge}
                                </span>
                            </div>
                            <div>
                                <h3 style={{ fontFamily: 'var(--font-heading)', fontSize: '1.25rem', fontWeight: 700, color: 'var(--color-text-primary)' }}>
                                    {feature.title}
                                </h3>
                                <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.875rem', fontWeight: 500, color: 'var(--color-brand-primary)', marginTop: '4px' }}>
                                    {feature.subtitle}
                                </p>
                            </div>
                            <p style={{ fontFamily: 'var(--font-body)', fontSize: '1rem', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
                                {feature.body}
                            </p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </section>
    );
};
