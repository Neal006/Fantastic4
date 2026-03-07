import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useRef, useEffect } from 'react';

const words = ['LUMIN', '⚡', 'AI', 'INTELLIGENCE', 'PLATFORM'];

export const HeroSection = () => {
    const sectionRef = useRef<HTMLElement>(null);
    const glowRef = useRef<HTMLDivElement>(null);
    const mouseX = useMotionValue(0);
    const mouseY = useMotionValue(0);

    // Smooth spring glow that trails the cursor
    const glowX = useSpring(mouseX, { stiffness: 60, damping: 20 });
    const glowY = useSpring(mouseY, { stiffness: 60, damping: 20 });

    useEffect(() => {
        const section = sectionRef.current;
        if (!section) return;
        const onMove = (e: MouseEvent) => {
            const rect = section.getBoundingClientRect();
            mouseX.set(e.clientX - rect.left);
            mouseY.set(e.clientY - rect.top);
        };
        section.addEventListener('mousemove', onMove);
        return () => section.removeEventListener('mousemove', onMove);
    }, [mouseX, mouseY]);

    return (
        <section
            ref={sectionRef}
            className="relative min-h-[100vh] w-full overflow-hidden flex flex-col items-center justify-center pt-24"
            style={{ backgroundColor: 'var(--color-bg-primary)' }}
        >
            {/* Background radial gold glow */}
            <motion.div
                animate={{ scale: [1, 1.08, 1] }}
                transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                style={{
                    position: 'absolute',
                    width: '80vw',
                    height: '80vw',
                    bottom: '-30%',
                    background: 'radial-gradient(ellipse 60% 50% at 50% 70%, rgba(212,144,10,0.12) 0%, transparent 70%)',
                    pointerEvents: 'none',
                }}
            />

            {/* Cursor-tracking glow — follows mouse smoothly */}
            <motion.div
                style={{
                    position: 'absolute',
                    width: '500px',
                    height: '500px',
                    borderRadius: '50%',
                    pointerEvents: 'none',
                    background: 'radial-gradient(circle, rgba(212,144,10,0.13) 0%, transparent 65%)',
                    x: glowX,
                    y: glowY,
                    translateX: '-50%',
                    translateY: '-50%',
                    zIndex: 0,
                }}
            />

            {/* Diagonal SVG polygon shapes bottom-right */}
            <svg viewBox="0 0 400 600" style={{ position: 'absolute', bottom: -100, right: -100, width: '400px', opacity: 0.15, pointerEvents: 'none' }}>
                <polygon points="400,0 400,600 80,600" fill="var(--color-brand-muted)" />
            </svg>

            {/* Floating Particles */}
            {Array.from({ length: 30 }).map((_, i) => (
                <motion.div
                    key={i}
                    initial={{ y: 0, opacity: 0.4 }}
                    animate={{ y: [0, -120], opacity: [0.4, 0] }}
                    transition={{
                        duration: 3 + Math.random() * 4,
                        repeat: Infinity,
                        delay: Math.random() * 4,
                        ease: "linear"
                    }}
                    style={{
                        position: 'absolute',
                        left: `${Math.random() * 100}%`,
                        top: `${Math.random() * 100}%`,
                        width: `${2 + Math.random() * 2}px`,
                        height: `${2 + Math.random() * 2}px`,
                        borderRadius: '50%',
                        background: 'var(--color-brand-light)',
                        pointerEvents: 'none',
                    }}
                />
            ))}

            {/* Hero Title */}
            <div className="relative z-10 text-center max-w-5xl px-4 flex flex-col items-center">
                <div className="flex flex-wrap justify-center items-center gap-x-4 gap-y-2 md:gap-x-6">
                    {words.slice(0, 3).map((w, i) => (
                        <motion.span
                            key={i}
                            initial={{ opacity: 0, y: 60 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: i * 0.08, duration: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
                            style={{
                                fontFamily: 'var(--font-display)',
                                fontSize: 'var(--text-hero)',
                                color: w === '⚡' ? 'var(--color-brand-accent)' : 'var(--color-text-primary)',
                                lineHeight: 1,
                            }}
                        >
                            {w}
                        </motion.span>
                    ))}
                </div>
                <motion.div
                    initial={{ opacity: 0, y: 60 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 3 * 0.08, duration: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
                    style={{
                        fontFamily: 'var(--font-display)',
                        fontSize: 'var(--text-hero)',
                        WebkitTextStroke: '2px var(--color-brand-primary)',
                        color: 'transparent',
                        lineHeight: 1,
                        marginTop: '-0.1em'
                    }}
                >
                    {words[3]}
                </motion.div>
                <motion.div
                    initial={{ opacity: 0, y: 60 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 4 * 0.08, duration: 0.7, ease: [0.25, 0.46, 0.45, 0.94] }}
                    style={{
                        fontFamily: 'var(--font-display)',
                        fontSize: 'var(--text-hero)',
                        color: 'var(--color-brand-primary)',
                        lineHeight: 1,
                        marginTop: '-0.1em'
                    }}
                >
                    {words[4]}
                </motion.div>

                {/* Subtitle label */}
                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6, duration: 0.7 }}
                    style={{
                        fontFamily: 'var(--font-body)',
                        fontSize: 'clamp(0.9rem, 1.8vw, 1.1rem)',
                        color: 'var(--color-text-secondary)',
                        marginTop: '2rem',
                        maxWidth: '520px',
                        lineHeight: 1.6,
                        letterSpacing: '0.01em',
                    }}
                >
                    AI-powered solar inverter fault prediction.
                    <br />
                    Prevent failures before they happen.
                </motion.p>
            </div>
        </section>
    );
};

