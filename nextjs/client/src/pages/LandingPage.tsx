import { useEffect, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { LoaderScreen } from '@/components/landing/LoaderScreen';
import { Navbar } from '@/components/landing/Navbar';
import { HeroSection } from '@/components/landing/HeroSection';
import { ProblemSection } from '@/components/landing/ProblemSection';
import { SolutionSection } from '@/components/landing/SolutionSection';
import { StorySection } from '@/components/landing/StorySection';
import { MetricsSection } from '@/components/landing/MetricsSection';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { MagneticButton } from '@/components/ui/Motion';
import { useNavigate } from 'react-router-dom';
import { Sun } from 'lucide-react';

gsap.registerPlugin(ScrollTrigger);

export default function LandingPage() {
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    // Background color interpolation
    useEffect(() => {
        if (loading) return;

        // Lerp background from cream (Hero) to gold (Problem)
        const st = ScrollTrigger.create({
            trigger: '#problem-section',
            start: 'top 80%',
            end: 'top 20%',
            scrub: true,
            onUpdate: (self) => {
                const progress = self.progress;
                const r = Math.round(255 + (212 - 255) * progress);
                const g = Math.round(253 + (144 - 253) * progress);
                const b = Math.round(245 + (10 - 245) * progress);
                document.body.style.background = `rgb(${r},${g},${b})`;
            }
        });

        return () => st.kill();
    }, [loading]);

    // Cursor Glow
    useEffect(() => {
        if (loading) return;
        const isTouchDevice = window.matchMedia('(hover: none)').matches;
        if (isTouchDevice) return;

        const glow = document.getElementById('cursor-glow');
        if (!glow) return;

        let mouseX = window.innerWidth / 2;
        let mouseY = window.innerHeight / 2;
        let glowX = mouseX;
        let glowY = mouseY;
        let animId: number;

        const onMove = (e: MouseEvent) => { mouseX = e.clientX; mouseY = e.clientY; };
        window.addEventListener('mousemove', onMove);

        const tick = () => {
            glowX += (mouseX - glowX) * 0.06;
            glowY += (mouseY - glowY) * 0.06;
            glow.style.transform = `translate(${glowX - 150}px, ${glowY - 150}px)`;
            animId = requestAnimationFrame(tick);
        };
        animId = requestAnimationFrame(tick);

        return () => {
            window.removeEventListener('mousemove', onMove);
            cancelAnimationFrame(animId);
        };
    }, [loading]);

    return (
        <>
            <AnimatePresence mode="wait">
                {loading ? (
                    <LoaderScreen key="loader" onComplete={() => setLoading(false)} />
                ) : (
                    <motion.div
                        key="content"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 0.8 }}
                    >
                        {/* Soft cursor glow */}
                        <div id="cursor-glow" style={{
                            position: 'fixed', width: '300px', height: '300px',
                            borderRadius: '50%', pointerEvents: 'none', zIndex: 9998,
                            background: 'radial-gradient(circle, rgba(212,144,10,0.05) 0%, transparent 70%)',
                            top: 0, left: 0,
                        }} />

                        <Navbar />
                        <HeroSection />
                        <ProblemSection />
                        <SolutionSection />
                        <StorySection />
                        <MetricsSection />

                        {/* CTA Footer */}
                        <section className="py-32 px-6 flex flex-col items-center justify-center text-center" style={{ backgroundColor: 'var(--color-bg-secondary)' }}>
                            <Sun className="h-16 w-16 mb-6" style={{ color: 'var(--color-brand-primary)' }} />
                            <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 'clamp(3rem, 8vw, 6rem)', color: 'var(--color-text-primary)' }}>
                                READY TO SEE THE FUTURE?
                            </h2>
                            <p style={{ fontFamily: 'var(--font-body)', fontSize: 'var(--text-h3)', color: 'var(--color-text-secondary)', maxWidth: '600px', margin: '24px 0 40px' }}>
                                Stop reacting to alarms. Start predicting faults. Enter the Lumin AI portal to view your plant intelligence.
                            </p>
                            <MagneticButton
                                onClick={() => navigate('/login')}
                                style={{
                                    padding: '20px 48px',
                                    borderRadius: 'var(--radius-sm)',
                                    background: 'var(--color-brand-primary)',
                                    color: 'var(--color-text-inverse)',
                                    fontFamily: 'var(--font-display)',
                                    fontSize: '2rem',
                                    letterSpacing: '0.05em',
                                    border: 'none',
                                    cursor: 'pointer',
                                    boxShadow: '0 12px 32px var(--color-shadow)',
                                }}
                            >
                                ENTER PORTAL
                            </MagneticButton>
                        </section>
                    </motion.div>
                )}
            </AnimatePresence>
        </>
    );
}
