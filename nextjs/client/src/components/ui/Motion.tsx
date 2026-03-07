import React, { useRef } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

export const FadeInView = ({ children, delay = 0, direction = 'up', className }: { children: React.ReactNode; delay?: number; direction?: 'up' | 'down' | 'left' | 'right' | 'scale'; className?: string }) => {
    const variants = {
        up: { hidden: { opacity: 0, y: 40 }, visible: { opacity: 1, y: 0 } },
        down: { hidden: { opacity: 0, y: -40 }, visible: { opacity: 1, y: 0 } },
        left: { hidden: { opacity: 0, x: -40 }, visible: { opacity: 1, x: 0 } },
        right: { hidden: { opacity: 0, x: 40 }, visible: { opacity: 1, x: 0 } },
        scale: { hidden: { opacity: 0, scale: 0.9 }, visible: { opacity: 1, scale: 1 } },
    };
    return (
        <motion.div
            className={className}
            variants={variants[direction]}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.2 }}
            transition={{ duration: 0.65, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
        >
            {children}
        </motion.div>
    );
};

export const MagneticButton = ({ children, style, onClick, className, ...props }: any) => {
    const ref = useRef<HTMLButtonElement>(null);
    const x = useMotionValue(0);
    const y = useMotionValue(0);
    const springX = useSpring(x, { stiffness: 300, damping: 20 });
    const springY = useSpring(y, { stiffness: 300, damping: 20 });

    const handleMove = (e: React.MouseEvent) => {
        if (!ref.current) return;
        const rect = ref.current.getBoundingClientRect();
        x.set((e.clientX - rect.left - rect.width / 2) * 0.25);
        y.set((e.clientY - rect.top - rect.height / 2) * 0.25);
    };

    return (
        <motion.button
            ref={ref}
            style={{ x: springX, y: springY, ...style }}
            className={className}
            onMouseMove={handleMove}
            onMouseLeave={() => { x.set(0); y.set(0); }}
            whileHover={{ scale: 1.06 }}
            whileTap={{ scale: 0.96 }}
            onClick={onClick}
            {...props}
        >
            {children}
        </motion.button>
    );
};
