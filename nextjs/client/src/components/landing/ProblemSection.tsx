import { useEffect } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SlashDivider } from '@/components/ui/Shapes';

gsap.registerPlugin(ScrollTrigger);

const stats = [
    { value: 4.6, unit: 'B', prefix: '$', label: 'Lost to inverter underperformance globally (2023)', icon: '💸' },
    { value: 17, unit: '%', prefix: '', label: 'Of all solar PV farm fault incidents caused by inverters', icon: '⚡' },
    { value: 80, unit: ' days', prefix: '', label: 'Average downtime per major inverter failure', icon: '🕐' },
    { value: 36, unit: '%', prefix: '', label: 'Of total plant energy losses caused by inverter faults (NREL)', icon: '📉' },
];

export const ProblemSection = () => {
    useEffect(() => {
        const ctx = gsap.context(() => {
            stats.forEach((stat, i) => {
                const underline = document.getElementById(`underline-${i}`);

                ScrollTrigger.create({
                    trigger: `#stat-${i}`,
                    start: 'top 80%',
                    once: true,
                    onEnter: () => {
                        // Animate counter
                        const obj = { val: 0 };
                        gsap.to(obj, {
                            val: stat.value,
                            duration: 2,
                            ease: 'power2.out',
                            onUpdate: () => {
                                const el = document.getElementById(`counter-${i}`);
                                if (el) {
                                    const numStr = obj.val.toFixed(stat.value % 1 !== 0 ? 1 : 0);
                                    el.textContent = stat.prefix + numStr + stat.unit;
                                }
                            }
                        });
                        // Animate underline width
                        if (underline) {
                            gsap.fromTo(underline,
                                { width: 0 },
                                { width: 48, duration: 0.6, ease: 'power2.out', delay: 0.3 }
                            );
                        }
                    }
                });
            });
        });

        return () => ctx.revert();
    }, []);

    return (
        <section id="problem-section" className="relative w-full py-24 pb-32" style={{ backgroundColor: 'var(--color-brand-primary)' }}>
            <SlashDivider color="var(--color-bg-primary)" flip={true} />

            {/* Watermark */}
            <div style={{
                position: 'absolute',
                top: '40%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                fontFamily: 'var(--font-display)',
                fontSize: '20vw',
                color: 'var(--color-brand-muted)',
                opacity: 0.08,
                pointerEvents: 'none',
                zIndex: 0,
                whiteSpace: 'nowrap'
            }}>
                IMPACT
            </div>

            <div className="relative z-10 max-w-5xl mx-auto px-6 mt-16 text-center lg:text-left">
                <h2 style={{
                    fontFamily: 'var(--font-display)',
                    fontSize: 'var(--text-display)',
                    color: 'var(--color-text-inverse)',
                    marginBottom: '3rem',
                    lineHeight: 1
                }}>
                    THE COST OF SILENCE
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-12 lg:gap-16">
                    {stats.map((stat, i) => (
                        <div key={i} id={`stat-${i}`} className="flex flex-col items-center md:items-start text-center md:text-left">
                            <div className="text-4xl mb-4">{stat.icon}</div>
                            <div
                                id={`counter-${i}`}
                                style={{
                                    fontFamily: 'var(--font-display)',
                                    fontSize: '5rem',
                                    color: 'var(--color-text-inverse)',
                                    lineHeight: 1,
                                    display: 'flex',
                                    alignItems: 'baseline'
                                }}
                            >
                                {stat.prefix}0{stat.unit}
                            </div>

                            <div
                                id={`underline-${i}`}
                                style={{
                                    height: '4px',
                                    width: '0px',
                                    backgroundColor: 'var(--color-brand-light)',
                                    margin: '16px 0',
                                    borderRadius: '2px'
                                }}
                            />

                            <p style={{
                                fontFamily: 'var(--font-body)',
                                fontSize: 'var(--text-h3)',
                                color: 'var(--color-text-inverse)',
                                opacity: 0.85,
                                maxWidth: '360px'
                            }}>
                                {stat.label}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};
