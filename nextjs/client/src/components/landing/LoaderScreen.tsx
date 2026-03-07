import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import gsap from 'gsap';

export const LoaderScreen = ({ onComplete }: { onComplete: () => void }) => {
    const [progress, setProgress] = useState(0);

    useEffect(() => {
        const obj = { val: 0 };
        const tl = gsap.timeline({
            onComplete: () => {
                // small hold at 100% before exit
                setTimeout(onComplete, 300);
            }
        });
        tl.to(obj, {
            val: 100,
            duration: 2.4,
            ease: 'power2.inOut',
            onUpdate: () => setProgress(Math.round(obj.val)),
        });

        return () => {
            tl.kill();
        };
    }, [onComplete]);

    return (
        <motion.div
            key="loader"
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.5, ease: 'easeInOut' }}
            style={{
                position: 'fixed', inset: 0, zIndex: 9999,
                background: 'var(--color-bg-secondary)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
        >
            {/* Close / skip button — top right */}
            <button
                onClick={onComplete}
                style={{
                    position: 'absolute', top: 20, right: 24,
                    background: 'none', border: 'none',
                    fontSize: '20px', cursor: 'pointer',
                    color: 'var(--color-text-primary)',
                    fontFamily: 'var(--font-body)',
                    opacity: 0.5,
                }}
                aria-label="Skip loader"
            >
                ✕
            </button>

            {/* Counter */}
            <span style={{
                fontFamily: 'var(--font-display)',
                fontSize: '96px',
                color: 'var(--color-text-primary)',
                lineHeight: 1,
            }}>
                {progress}%
            </span>

            {/* Progress bar — bottom */}
            <div style={{
                position: 'absolute', bottom: 0, left: 0,
                width: '100%', height: '3px',
                background: 'var(--color-bg-tertiary)',
            }}>
                <motion.div style={{
                    height: '100%',
                    width: `${progress}%`,
                    background: 'var(--color-brand-primary)',
                }} />
            </div>
        </motion.div>
    );
};
