import { motion } from 'framer-motion';

export const BgPolygon = ({ style }: { style?: React.CSSProperties }) => (
    <svg viewBox="0 0 400 600" style={{ position: 'absolute', pointerEvents: 'none', ...style }}>
        <polygon points="400,0 400,600 80,600" fill="var(--color-brand-muted)" />
    </svg>
);

export const SlashDivider = ({ flip = false, color = 'var(--color-brand-primary)' }: { flip?: boolean; color?: string }) => (
    <svg viewBox="0 0 1440 80" style={{ display: 'block', marginTop: '-1px', transform: flip ? 'scaleX(-1)' : 'none' }}>
        <polygon points="0,0 1440,0 1440,80" fill={color} />
    </svg>
);

export const RotatingDiamond = ({ size = 40, speed = 20, style }: { size?: number; speed?: number; style?: React.CSSProperties }) => (
    <motion.div
        style={{ ...style }}
        animate={{ rotate: 360 }}
        transition={{ duration: speed, repeat: Infinity, ease: 'linear' }}
    >
        <svg viewBox="0 0 40 40" width={size} height={size}>
            <polygon
                points="20,2 38,20 20,38 2,20"
                fill="none"
                stroke="var(--color-brand-light)"
                strokeWidth="1.5"
            />
        </svg>
    </motion.div>
);
