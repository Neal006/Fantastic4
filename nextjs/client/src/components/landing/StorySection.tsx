import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const storyPanels = [
    {
        bg: 'var(--color-brand-primary)', text: 'var(--color-text-inverse)', num: '01',
        headline: 'Jaw-Dropping Financial Losses',
        body: (
            <div className="space-y-4">
                <p>Solar asset underperformance, driven heavily by inverters, cost the global industry a record $10 billion in lost revenue in 2024 alone.</p>
                <ul className="list-disc pl-5 space-y-2 lg:space-y-3">
                    <li><strong>Massive Power Deficits:</strong> In the U.S., facilities lost $5,720 per MW to equipment issues, with inverters causing 39% of power losses—over 50% in key markets like ERCOT and MISO.</li>
                    <li><strong>Spiking O&M Costs:</strong> For a 100 MW plant, single inverter failure types can exceed €3.8 million/year in losses, driving operation and maintenance costs up by 23%.</li>
                </ul>
            </div>
        ),
    },
    {
        bg: 'var(--color-bg-secondary)', text: 'var(--color-text-primary)', num: '02',
        headline: 'Alarming Failure Rates',
        body: (
            <div className="space-y-4">
                <p>Inverters cause 43-60% of all PV plant failures, far outpacing modules, which account for just 5% of failures.</p>
                <ul className="list-disc pl-5 space-y-2 lg:space-y-3">
                    <li><strong>Rapid Degradation:</strong> 45% of solar inverters fail within 4 years, per LBNL's 2024 reliability study, with central inverters hitting a staggering 52% failure rate.</li>
                    <li><strong>Warranty Shortfalls:</strong> Lifetimes average 10-12 years while warranties typically only cover 5 years, fueling over 60% of PV system failures according to DOE SETO.</li>
                </ul>
            </div>
        ),
    },
    {
        bg: 'var(--color-brand-primary)', text: 'var(--color-text-inverse)', num: '03',
        headline: 'Brutal Downtime Impact',
        body: (
            <div className="space-y-4">
                <p>Inverter failures trigger 36% of total energy losses in solar plants, per NREL research conducted over a 27-month period.</p>
                <ul className="list-disc pl-5 space-y-2 lg:space-y-3">
                    <li><strong>Crippling Delays:</strong> Major breakdowns average 80 days of downtime, wiping out a quarter of annual production even during peak weather conditions.</li>
                    <li><strong>Maintenance Burden:</strong> 51% of O&M tickets target inverters (fans, controllers, stacks). While repairs dominate the workload, full replacements cost significantly more.</li>
                </ul>
            </div>
        ),
    },
    {
        bg: 'var(--color-bg-secondary)', text: 'var(--color-text-primary)', num: '04',
        headline: 'Industry on the Brink',
        body: (
            <div className="space-y-4">
                <p>U.S. insurers now flag inverters as top attritional claims, noting that central and string units often fail just 1.5 to 2.2 years post-install.</p>
                <ul className="list-disc pl-5 space-y-2 lg:space-y-3">
                    <li><strong>Supply & Labor Blockers:</strong> 45% of inverter capacity comes from defunct makers after 4 years, leaving no spares or techs, while labor shortages block 44% of industry growth.</li>
                    <li><strong>Stifling Scale:</strong> NREL warns that without fixes, short inverter life will stifle solar's scale-up amidst rising energy demand and the shift from 1500V to 2000V systems.</li>
                </ul>
            </div>
        ),
    },
];

export const StorySection = () => {
    const trackRef = useRef<HTMLDivElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    // Fallback for mobile
    const isMobile = typeof window !== 'undefined' && window.innerWidth < 768;

    useEffect(() => {
        if (isMobile) return;

        const panels = gsap.utils.toArray<HTMLElement>('.story-panel');
        const totalWidth = panels.length * window.innerWidth;

        // Pre-hide all panel content first
        panels.forEach((panel) => {
            const headline = panel.querySelector('.panel-headline');
            const body = panel.querySelector('.panel-body');
            gsap.set(headline, { opacity: 0, y: 40 });
            gsap.set(body, { opacity: 0, y: 30 });
        });

        const ctx = gsap.context(() => {
            // ── Horizontal scroll tween ──
            const tween = gsap.to(trackRef.current, {
                x: () => -(totalWidth - window.innerWidth),
                ease: 'none',
                scrollTrigger: {
                    trigger: containerRef.current,
                    pin: true,
                    scrub: 1.2,
                    snap: {
                        snapTo: 1 / (panels.length - 1),
                        duration: { min: 0.3, max: 0.6 },
                        ease: 'power2.inOut',
                    },
                    end: () => `+=${totalWidth - window.innerWidth}`,
                    invalidateOnRefresh: true,
                    onUpdate: (self) => {
                        // Progress goes 0→1 over the full scroll.
                        // Each panel covers 1/n of the progress range.
                        const n = panels.length;
                        panels.forEach((panel, i) => {
                            const headline = panel.querySelector('.panel-headline');
                            const body = panel.querySelector('.panel-body');
                            const panelStart = i / n;
                            const panelRevealAt = panelStart + 0.02; // small offset so first panel shows fast
                            if (self.progress >= panelRevealAt) {
                                gsap.to(headline, { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out', overwrite: 'auto' });
                                gsap.to(body, { opacity: 1, y: 0, duration: 0.5, delay: 0.08, ease: 'power3.out', overwrite: 'auto' });
                            } else {
                                // Partially scrolled back — hide again
                                if (i > 0) {
                                    gsap.to(headline, { opacity: 0, y: 40, duration: 0.3, overwrite: 'auto' });
                                    gsap.to(body, { opacity: 0, y: 30, duration: 0.3, overwrite: 'auto' });
                                }
                            }
                        });
                    },
                    onEnter: () => {
                        // Reveal first panel instantly on scroll enter
                        const headline = panels[0]?.querySelector('.panel-headline');
                        const body = panels[0]?.querySelector('.panel-body');
                        gsap.to(headline, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out' });
                        gsap.to(body, { opacity: 1, y: 0, duration: 0.7, delay: 0.1, ease: 'power3.out' });
                    },
                },
            });
        }, containerRef);

        return () => { ctx.revert(); };
    }, [isMobile]);

    if (isMobile) {
        return (
            <div className="flex flex-col">
                {storyPanels.map((panel, i) => (
                    <div key={i} style={{ backgroundColor: panel.bg, color: panel.text, padding: '80px 24px', position: 'relative' }}>
                        <div style={{
                            position: 'absolute', top: 20, right: 20,
                            fontFamily: 'var(--font-display)', fontSize: '20vw', color: panel.text, opacity: 0.06, lineHeight: 1
                        }}>
                            {panel.num}
                        </div>
                        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(3rem, 12vw, 4rem)', marginBottom: '16px', lineHeight: 1 }}>
                            {panel.headline}
                        </h2>
                        <div style={{ fontFamily: 'var(--font-body)', fontSize: '1rem', lineHeight: 1.6, opacity: 0.9 }}>
                            {panel.body}
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    return (
        <div ref={containerRef} style={{ position: 'relative' }}>
            <div
                ref={trackRef}
                style={{
                    display: 'flex',
                    width: `${storyPanels.length * 100}vw`,
                    willChange: 'transform',
                }}
            >
                {storyPanels.map((panel, i) => (
                    <div
                        key={i}
                        className="story-panel"
                        style={{
                            width: '100vw',
                            height: '100vh',
                            background: panel.bg,
                            color: panel.text,
                            flexShrink: 0,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            padding: '0 10vw',
                            position: 'relative',
                            overflow: 'hidden'
                        }}
                    >
                        {/* Number Watermark */}
                        <div style={{
                            position: 'absolute',
                            top: '5%',
                            right: '5%',
                            fontFamily: 'var(--font-display)',
                            fontSize: '25vw',
                            opacity: 0.06,
                            pointerEvents: 'none',
                            lineHeight: 1
                        }}>
                            {panel.num}
                        </div>

                        <div className="max-w-4xl w-full">
                            <h2 className="panel-headline" style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: 'clamp(3rem, 6vw, 6rem)',
                                marginBottom: '1.5rem',
                                lineHeight: 1,
                            }}>
                                {panel.headline}
                            </h2>
                            <div className="panel-body" style={{
                                fontFamily: 'var(--font-body)',
                                fontSize: 'clamp(1rem, 1.3vw, 1.125rem)',
                                maxWidth: '800px',
                                lineHeight: 1.6,
                                opacity: 0.9
                            }}>
                                {panel.body}
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
