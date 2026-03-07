import { useEffect, useRef, useState } from 'react';
import { motion, useInView, animate } from 'framer-motion';
import {
    Leaf, Zap, Thermometer, Sun, BarChart3, Globe,
} from 'lucide-react';

/* ─────────────────────────────────────────────────────────
   Types
───────────────────────────────────────────────────────── */
interface KpiData {
    vcoy: { tonnes: number; carbon_revenue_usd: number; energy_kwh_30d: number };
    elf: { loss_kwh: number; revenue_loss_usd: number };
    tdi: { score: number; health_pct: number };
    sue: { score: number; percent: number };
    ppr: { score: number; percent: number };
    eis: { score: number; percent: number };
    fleet: { total_inverters: number; healthy_pct: number; faulty_count: number };
}

/* ─────────────────────────────────────────────────────────
   KPI definitions
───────────────────────────────────────────────────────── */
function buildCards(data: KpiData) {
    return [
        {
            id: 'vcoy',
            icon: <Leaf size={32} />,
            acronym: 'VCOY',
            label: 'Verified Carbon Offset Yield',
            accent: '#22c55e',
            value: data.vcoy.tonnes,
            unit: 'tonnes CO₂',
            sub: `$${data.vcoy.carbon_revenue_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })} carbon credits earned`,
            detail: `${data.vcoy.energy_kwh_30d.toLocaleString()} kWh generated (30d)`,
            gauge: Math.min(data.vcoy.tonnes / 200, 1),   // normalise for visual
        },
        {
            id: 'elf',
            icon: <Zap size={32} />,
            acronym: 'ELF',
            label: 'Energy Loss Due to Faults',
            accent: '#ef4444',
            value: data.elf.loss_kwh,
            unit: 'kWh lost',
            sub: `$${data.elf.revenue_loss_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })} revenue leakage`,
            detail: 'Caused by faulty inverters (30d)',
            gauge: Math.min(data.elf.loss_kwh / 5000, 1),
        },
        {
            id: 'tdi',
            icon: <Thermometer size={32} />,
            acronym: 'TDI',
            label: 'Thermal Degradation Index',
            accent: '#f97316',
            value: data.tdi.score,
            unit: '/ 1.0',
            sub: `${data.tdi.health_pct}% thermal health`,
            detail: '0 = healthy · 1 = critical degradation',
            gauge: data.tdi.score,
        },
        {
            id: 'sue',
            icon: <Sun size={32} />,
            acronym: 'SUE',
            label: 'Solar Utilization Efficiency',
            accent: '#d4900a',
            value: data.sue.percent,
            unit: '%',
            sub: `Score: ${data.sue.score.toFixed(3)}`,
            detail: '80%+ normal · <60% signals degradation',
            gauge: data.sue.score,
        },
        {
            id: 'ppr',
            icon: <BarChart3 size={32} />,
            acronym: 'PPR',
            label: 'Plant Performance Ratio',
            accent: '#8b5cf6',
            value: data.ppr.percent,
            unit: '%',
            sub: `Score: ${data.ppr.score.toFixed(3)}`,
            detail: 'Actual ÷ Theoretical solar output',
            gauge: data.ppr.score,
        },
        {
            id: 'eis',
            icon: <Globe size={32} />,
            acronym: 'EIS',
            label: 'Environmental Impact Score',
            accent: '#06b6d4',
            value: data.eis.percent,
            unit: '%',
            sub: `Composite sustainability index`,
            detail: 'Offset · Efficiency · Downtime weighted',
            gauge: data.eis.score,
        },
    ];
}

/* ─────────────────────────────────────────────────────────
   Animated number component
───────────────────────────────────────────────────────── */
function CountUp({ to, decimals = 0 }: { to: number; decimals?: number }) {
    const ref = useRef<HTMLSpanElement>(null);
    const inView = useInView(ref, { once: true, margin: '-80px' });

    useEffect(() => {
        if (!inView) return;
        const controls = animate(0, to, {
            duration: 1.8,
            ease: [0.25, 0.46, 0.45, 0.94],
            onUpdate(v) {
                if (ref.current) ref.current.textContent = v.toFixed(decimals);
            },
        });
        return controls.stop;
    }, [inView, to, decimals]);

    return <span ref={ref}>0</span>;
}

/* ─────────────────────────────────────────────────────────
   Circular gauge SVG
───────────────────────────────────────────────────────── */
function CircleGauge({ value, color }: { value: number; color: string }) {
    const r = 36;
    const circumference = 2 * Math.PI * r;
    const clampedValue = Math.max(0, Math.min(value, 1));
    const offset = circumference * (1 - clampedValue);
    const ref = useRef<SVGCircleElement>(null);
    const inView = useInView(ref, { once: true, margin: '-40px' });

    useEffect(() => {
        if (!inView || !ref.current) return;
        // Animate stroke-dashoffset from full to target
        const el = ref.current;
        el.style.transition = 'none';
        el.style.strokeDashoffset = String(circumference);
        // Force reflow
        void el.getBoundingClientRect();
        el.style.transition = 'stroke-dashoffset 1.8s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
        el.style.strokeDashoffset = String(offset);
    }, [inView, offset, circumference]);

    return (
        <svg width="90" height="90" viewBox="0 0 90 90" style={{ transform: 'rotate(-90deg)' }}>
            {/* Track */}
            <circle cx="45" cy="45" r={r} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="6" />
            {/* Progress */}
            <circle
                ref={ref}
                cx="45" cy="45" r={r}
                fill="none"
                stroke={color}
                strokeWidth="6"
                strokeLinecap="round"
                strokeDasharray={circumference}
                strokeDashoffset={circumference}
                style={{ willChange: 'stroke-dashoffset' }}
            />
        </svg>
    );
}

/* ─────────────────────────────────────────────────────────
   Individual Card
───────────────────────────────────────────────────────── */
function KpiCard({ card, index }: { card: ReturnType<typeof buildCards>[0]; index: number }) {
    const isNegative = card.id === 'elf';   // red / "bad" metric
    const decimals = card.unit === '/ 1.0' ? 3 : (card.unit === 'tonnes CO₂' ? 2 : 0);

    return (
        <motion.div
            initial={{ opacity: 0, y: 60 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.15 }}
            transition={{ duration: 0.7, delay: index * 0.1, ease: [0.25, 0.46, 0.45, 0.94] }}
            whileHover={{ y: -8, boxShadow: `0 24px 60px ${card.accent}30` }}
            style={{
                background: 'rgba(255,248,231,0.04)',
                border: `1px solid ${card.accent}30`,
                borderRadius: '20px',
                padding: '28px',
                backdropFilter: 'blur(20px)',
                position: 'relative',
                overflow: 'hidden',
                cursor: 'default',
            }}
        >
            {/* Glow blob */}
            <div style={{
                position: 'absolute', top: '-40px', right: '-40px',
                width: '140px', height: '140px', borderRadius: '50%',
                background: `radial-gradient(circle, ${card.accent}22 0%, transparent 70%)`,
                pointerEvents: 'none',
            }} />

            {/* Header row */}
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: '20px' }}>
                <div>
                    {/* Badge */}
                    <span style={{
                        display: 'inline-block',
                        fontSize: '0.65rem', fontWeight: 700, fontFamily: 'var(--font-sans)',
                        letterSpacing: '0.12em', padding: '3px 10px', borderRadius: '20px',
                        background: `${card.accent}22`, color: card.accent,
                        marginBottom: '10px',
                    }}>
                        {card.acronym}
                    </span>
                    <div style={{ color: card.accent, marginTop: '4px' }}>
                        {card.icon}
                    </div>
                </div>
                {/* Circular gauge */}
                <div style={{ position: 'relative', width: '90px', height: '90px' }}>
                    <CircleGauge value={card.gauge} color={card.accent} />
                    <div style={{
                        position: 'absolute', inset: 0, display: 'flex',
                        alignItems: 'center', justifyContent: 'center',
                        fontSize: '0.7rem', fontWeight: 700,
                        color: card.accent, fontFamily: 'var(--font-sans)',
                    }}>
                        {Math.round(card.gauge * 100)}%
                    </div>
                </div>
            </div>

            {/* Label */}
            <p style={{
                fontFamily: 'var(--font-body)', fontSize: '0.8rem',
                color: 'rgba(255,255,255,0.5)', marginBottom: '6px',
            }}>
                {card.label}
            </p>

            {/* Big number */}
            <div style={{
                fontFamily: 'var(--font-display)', fontSize: 'clamp(2rem, 3.5vw, 3rem)',
                color: '#FFFDF5', lineHeight: 1, marginBottom: '6px',
                display: 'flex', alignItems: 'baseline', gap: '6px',
            }}>
                <CountUp to={card.value} decimals={decimals} />
                <span style={{ fontSize: '0.9rem', opacity: 0.6, fontFamily: 'var(--font-body)' }}>
                    {card.unit}
                </span>
            </div>

            {/* Divider */}
            <motion.div
                initial={{ scaleX: 0 }}
                whileInView={{ scaleX: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8, delay: 0.3 + index * 0.08 }}
                style={{
                    height: '1px', background: `linear-gradient(90deg, ${card.accent}80, transparent)`,
                    transformOrigin: 'left', marginBottom: '12px', marginTop: '12px',
                }}
            />

            {/* Sub info */}
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.85rem', color: 'rgba(255,255,255,0.75)', marginBottom: '4px' }}>
                {card.sub}
            </p>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.75rem', color: 'rgba(255,255,255,0.4)' }}>
                {card.detail}
            </p>

            {/* Alert flash for ELF if significant losses */}
            {isNegative && card.value > 100 && (
                <motion.div
                    animate={{ opacity: [0.6, 1, 0.6] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    style={{
                        marginTop: '12px', padding: '6px 12px', borderRadius: '8px',
                        background: `${card.accent}20`, border: `1px solid ${card.accent}40`,
                        fontSize: '0.7rem', color: card.accent, fontFamily: 'var(--font-body)',
                        fontWeight: 600,
                    }}
                >
                    ⚠ Revenue leakage detected
                </motion.div>
            )}
        </motion.div>
    );
}

/* ─────────────────────────────────────────────────────────
   Main section
───────────────────────────────────────────────────────── */
export const MetricsSection = () => {
    const [data, setData] = useState<KpiData | null>(null);
    const [error, setError] = useState(false);

    useEffect(() => {
        fetch('/api/metrics/kpi')
            .then(r => r.json())
            .then(d => setData(d))
            .catch(() => setError(true));
    }, []);

    const cards = data ? buildCards(data) : null;

    return (
        <section
            id="impact"
            style={{
                background: '#0a0a0a',
                padding: 'clamp(80px, 12vw, 140px) 5%',
                position: 'relative',
            }}
        >
            {/* Background grid pattern */}
            <div style={{
                position: 'absolute', inset: 0, pointerEvents: 'none', opacity: 0.04,
                backgroundImage: 'repeating-linear-gradient(0deg, rgba(255,255,255,0.5) 0px, rgba(255,255,255,0.5) 1px, transparent 1px, transparent 60px), repeating-linear-gradient(90deg, rgba(255,255,255,0.5) 0px, rgba(255,255,255,0.5) 1px, transparent 1px, transparent 60px)',
            }} />

            {/* Ambient glow */}
            <motion.div
                animate={{ opacity: [0.2, 0.4, 0.2] }}
                transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
                style={{
                    position: 'absolute', top: '10%', left: '50%', transform: 'translateX(-50%)',
                    width: '80vw', height: '80vw', pointerEvents: 'none',
                    background: 'radial-gradient(ellipse, rgba(212,144,10,0.08) 0%, transparent 60%)',
                }}
            />

            <div style={{ maxWidth: '1400px', margin: '0 auto', position: 'relative', zIndex: 1 }}>

                {/* Section header */}
                <div style={{ textAlign: 'center', marginBottom: 'clamp(48px, 8vw, 80px)' }}>
                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5 }}
                        style={{
                            fontFamily: 'var(--font-body)', fontWeight: 700, fontSize: '0.8rem',
                            letterSpacing: '0.2em', color: 'var(--color-brand-primary)',
                            marginBottom: '16px', textTransform: 'uppercase',
                        }}
                    >
                        Live Platform Metrics
                    </motion.p>
                    <motion.h2
                        initial={{ opacity: 0, y: 30 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                        style={{
                            fontFamily: 'var(--font-display)',
                            fontSize: 'clamp(2.5rem, 6vw, 5.5rem)',
                            color: '#FFFDF5', lineHeight: 1, marginBottom: '20px',
                        }}
                    >
                        INTELLIGENCE
                        <br />
                        <span style={{ WebkitTextStroke: '2px var(--color-brand-primary)', color: 'transparent' }}>
                            BY THE NUMBERS
                        </span>
                    </motion.h2>
                    <motion.p
                        initial={{ opacity: 0 }}
                        whileInView={{ opacity: 1 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5, delay: 0.2 }}
                        style={{
                            fontFamily: 'var(--font-body)', fontSize: 'clamp(1rem, 2vw, 1.25rem)',
                            color: 'rgba(255,253,245,0.55)', maxWidth: '600px', margin: '0 auto',
                        }}
                    >
                        Six research-backed KPIs computed in real-time from live inverter telemetry.
                        Every number below is pulled directly from the database.
                    </motion.p>
                </div>

                {/* Fleet summary bar */}
                {data && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.5 }}
                        style={{
                            display: 'flex', gap: '2px',
                            justifyContent: 'center', flexWrap: 'wrap',
                            marginBottom: '60px',
                        }}
                    >
                        {[
                            { label: 'Total Inverters', value: data.fleet.total_inverters, color: '#FFFDF5' },
                            { label: 'Fleet Health', value: `${data.fleet.healthy_pct}%`, color: '#22c55e' },
                            { label: 'Fault / Offline', value: data.fleet.faulty_count, color: '#ef4444' },
                        ].map((stat, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, scale: 0.9 }}
                                whileInView={{ opacity: 1, scale: 1 }}
                                viewport={{ once: true }}
                                transition={{ delay: 0.2 + i * 0.1 }}
                                style={{
                                    padding: '12px 32px',
                                    background: 'rgba(255,248,231,0.05)',
                                    border: '1px solid rgba(255,248,231,0.1)',
                                    borderRadius: i === 0 ? '40px 0 0 40px' : i === 2 ? '0 40px 40px 0' : '0',
                                    textAlign: 'center', minWidth: '160px',
                                }}
                            >
                                <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.8rem', color: stat.color }}>
                                    {typeof stat.value === 'number' ? (
                                        <CountUp to={stat.value} />
                                    ) : stat.value}
                                </div>
                                <div style={{ fontFamily: 'var(--font-body)', fontSize: '0.7rem', color: 'rgba(255,255,255,0.4)', letterSpacing: '0.1em', marginTop: '2px' }}>
                                    {stat.label.toUpperCase()}
                                </div>
                            </motion.div>
                        ))}
                    </motion.div>
                )}

                {/* Loading state */}
                {!data && !error && (
                    <div style={{ textAlign: 'center', padding: '80px 0' }}>
                        <motion.div
                            animate={{ rotate: 360 }}
                            transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                            style={{
                                width: '48px', height: '48px', borderRadius: '50%',
                                border: '3px solid rgba(212,144,10,0.2)',
                                borderTop: '3px solid var(--color-brand-primary)',
                                display: 'inline-block',
                            }}
                        />
                        <p style={{ fontFamily: 'var(--font-body)', color: 'rgba(255,255,255,0.4)', marginTop: '20px' }}>
                            Computing live metrics…
                        </p>
                    </div>
                )}

                {/* Error state */}
                {error && (
                    <div style={{
                        textAlign: 'center', padding: '80px 0',
                        color: 'rgba(255,255,255,0.4)', fontFamily: 'var(--font-body)',
                    }}>
                        Unable to fetch live data. Metrics shown once backend is connected.
                    </div>
                )}

                {/* KPI Grid */}
                {cards && (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(min(340px, 100%), 1fr))',
                        gap: '20px',
                    }}>
                        {cards.map((card, i) => (
                            <KpiCard key={card.id} card={card} index={i} />
                        ))}
                    </div>
                )}

                {/* Bottom note */}
                <motion.p
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.5 }}
                    style={{
                        textAlign: 'center', marginTop: '48px',
                        fontFamily: 'var(--font-body)', fontSize: '0.75rem',
                        color: 'rgba(255,255,255,0.25)',
                    }}
                >
                    Metrics computed over the last 30 days from live inverter telemetry.
                    VCOY uses global avg grid emission factor 0.71 kg CO₂/kWh.
                    ELF assumes 96% inverter efficiency and $0.06/kWh solar PPA tariff.
                </motion.p>
            </div>
        </section>
    );
};
