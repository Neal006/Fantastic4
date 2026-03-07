import { useEffect, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Loader2 } from 'lucide-react';
import anime from 'animejs';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function useCountUp(target: number, triggered: boolean, duration = 1800) {
    const [value, setValue] = useState(0);
    useEffect(() => {
        if (!triggered || target === 0) return;
        const obj = { val: 0 };
        anime({
            targets: obj,
            val: target,
            duration,
            easing: 'easeOutExpo',
            update: () => setValue(Math.round(obj.val * 10) / 10),
        });
    }, [triggered, target, duration]);
    return value;
}

function useInView(threshold = 0.25) {
    const ref = useRef<HTMLDivElement>(null);
    const [inView, setInView] = useState(false);
    useEffect(() => {
        const el = ref.current;
        if (!el) return;
        const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setInView(true); }, { threshold });
        obs.observe(el);
        return () => obs.disconnect();
    }, [threshold]);
    return { ref, inView };
}

function fmt(n: number) { return n.toLocaleString('en-IN'); }
function fmtDec(n: number, d = 1) { return n.toFixed(d); }

// ─── SVG components ───────────────────────────────────────────────────────────

function SunSVG() {
    const raysRef = useRef<SVGGElement>(null);
    const circleRef = useRef<SVGCircleElement>(null);
    useEffect(() => {
        if (raysRef.current) {
            anime({
                targets: raysRef.current.querySelectorAll('line'),
                strokeDashoffset: [anime.setDashoffset, 0],
                easing: 'easeInOutSine',
                duration: 1200,
                delay: anime.stagger(80),
            });
        }
        if (circleRef.current) {
            anime({
                targets: circleRef.current,
                r: [0, 54],
                opacity: [0, 1],
                duration: 900,
                easing: 'easeOutElastic(1, .6)',
            });
        }
        // Continuous slow rotation
        anime({
            targets: raysRef.current,
            rotate: '1turn',
            duration: 28000,
            loop: true,
            easing: 'linear',
        });
        anime({
            targets: circleRef.current,
            scale: [1, 1.06, 1],
            duration: 3000,
            loop: true,
            easing: 'easeInOutSine',
        });
    }, []);

    const rays = Array.from({ length: 12 }, (_, i) => {
        const angle = (i * 30 * Math.PI) / 180;
        const x1 = 120 + Math.cos(angle) * 68;
        const y1 = 120 + Math.sin(angle) * 68;
        const x2 = 120 + Math.cos(angle) * 95;
        const y2 = 120 + Math.sin(angle) * 95;
        return { x1, y1, x2, y2 };
    });

    return (
        <svg viewBox="0 0 240 240" className="w-full h-full" style={{ filter: 'drop-shadow(0 0 24px #F5A62388)' }}>
            <defs>
                <radialGradient id="sunGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#FFE599" />
                    <stop offset="60%" stopColor="#F5A623" />
                    <stop offset="100%" stopColor="#D4900A" />
                </radialGradient>
                <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#FFD16640" />
                    <stop offset="100%" stopColor="#D4900A00" />
                </radialGradient>
            </defs>
            <circle cx="120" cy="120" r="90" fill="url(#glowGrad)" />
            <g ref={raysRef} style={{ transformOrigin: '120px 120px' }}>
                {rays.map((r, i) => (
                    <line key={i} x1={r.x1} y1={r.y1} x2={r.x2} y2={r.y2}
                        stroke="#F5A623" strokeWidth="5" strokeLinecap="round"
                        style={{ filter: 'drop-shadow(0 0 4px #FFD166)' }} />
                ))}
            </g>
            <circle ref={circleRef} cx="120" cy="120" r="54" fill="url(#sunGrad)" />
        </svg>
    );
}

function LeafSVG({ animated = false }: { animated?: boolean }) {
    const ref = useRef<SVGPathElement>(null);
    useEffect(() => {
        if (!animated || !ref.current) return;
        anime({
            targets: ref.current,
            strokeDashoffset: [anime.setDashoffset, 0],
            duration: 1500,
            easing: 'easeInOutQuart',
        });
    }, [animated]);
    return (
        <svg viewBox="0 0 80 80" className="w-10 h-10">
            <path ref={ref} d="M40 70 C10 50, 5 20, 40 10 C75 20, 70 50, 40 70Z"
                fill="#D4E6CF" stroke="#6B8F5E" strokeWidth="2" />
            <line x1="40" y1="10" x2="40" y2="70" stroke="#6B8F5E" strokeWidth="1.5" />
            <line x1="40" y1="30" x2="28" y2="22" stroke="#6B8F5E" strokeWidth="1" />
            <line x1="40" y1="40" x2="25" y2="35" stroke="#6B8F5E" strokeWidth="1" />
            <line x1="40" y1="50" x2="28" y2="48" stroke="#6B8F5E" strokeWidth="1" />
        </svg>
    );
}

function GaugeSVG({ percent, label, color, benchmarkLabel }: {
    percent: number; label: string; color: string; benchmarkLabel: string;
}) {
    const svgRef = useRef<SVGElement>(null);
    const [displayed, setDisplayed] = useState(0);
    const { ref, inView } = useInView();

    const r = 70; const cx = 90; const cy = 90;
    const circumference = Math.PI * r; // half circle
    const strokeDash = circumference;
    const offset = circumference - (percent / 100) * circumference;

    useEffect(() => {
        if (!inView) return;
        const obj = { p: 0 };
        anime({
            targets: obj,
            p: percent,
            easing: 'easeOutCubic',
            duration: 1600,
            update: () => setDisplayed(Math.round(obj.p * 10) / 10),
        });
        if (svgRef.current) {
            anime({
                targets: svgRef.current.querySelector('.gauge-fill'),
                strokeDashoffset: [circumference, offset],
                duration: 1600,
                easing: 'easeOutCubic',
            });
        }
    }, [inView, percent]);

    return (
        <div ref={ref} className="flex flex-col items-center gap-2">
            <svg ref={svgRef as any} viewBox="0 0 180 100" className="w-48">
                {/* Background arc */}
                <path d={`M 20 90 A ${r} ${r} 0 0 1 160 90`}
                    fill="none" stroke="#E8D5A3" strokeWidth="12" strokeLinecap="round" />
                {/* Fill arc */}
                <path className="gauge-fill"
                    d={`M 20 90 A ${r} ${r} 0 0 1 160 90`}
                    fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
                    strokeDasharray={`${circumference}`}
                    strokeDashoffset={circumference}
                    style={{ transition: 'none' }}
                />
                <text x="90" y="78" textAnchor="middle" fontSize="26" fontWeight="bold"
                    fill="#2C2C2C" fontFamily="Bebas Neue, sans-serif">
                    {displayed}%
                </text>
            </svg>
            <p className="font-semibold text-sm text-center" style={{ color: '#2C2C2C', fontFamily: 'Montserrat, sans-serif' }}>
                {label}
            </p>
            <span className="text-xs px-2 py-0.5 rounded-full font-medium"
                style={{ background: color + '22', color }}>
                {benchmarkLabel}
            </span>
        </div>
    );
}

// ─── Horizontal Scroll Section ────────────────────────────────────────────────

function HorizontalStory({ data }: { data: any }) {
    const trackRef = useRef<HTMLDivElement>(null);
    const wrapRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const wrap = wrapRef.current;
        const track = trackRef.current;
        if (!wrap || !track) return;

        const onScroll = () => {
            const rect = wrap.getBoundingClientRect();
            const progress = Math.max(0, Math.min(1, -rect.top / (wrap.offsetHeight - window.innerHeight)));
            const maxTranslate = track.scrollWidth - track.offsetWidth;
            track.style.transform = `translateX(-${progress * maxTranslate}px)`;
        };

        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    const cards = [
        {
            icon: '⚡',
            label: 'Energy Generated',
            value: `${fmt(data?.energy_kwh_30d ?? 0)} kWh`,
            sub: 'Last 30 days',
            color: '#D4900A',
            bg: '#FFF3CC',
        },
        {
            icon: '🌿',
            label: 'CO₂ Avoided',
            value: `${fmtDec(data?.co2_avoided_tonnes ?? 0)} tonnes`,
            sub: 'vs coal grid baseline',
            color: '#6B8F5E',
            bg: '#D4E6CF',
        },
        {
            icon: '🏅',
            label: 'Carbon Credits Earned',
            value: `${fmtDec(data?.carbon_credits ?? 0)}`,
            sub: '1 credit = 1 tonne CO₂',
            color: '#D4900A',
            bg: '#FFF8E7',
        },
        {
            icon: '💵',
            label: 'Credit Market Value',
            value: `$${fmt(data?.credit_value_usd ?? 0)}`,
            sub: '@$22/tonne (VCM rate)',
            color: '#6B8F5E',
            bg: '#D4E6CF',
        },
        {
            icon: '🌳',
            label: 'Equivalent Trees',
            value: fmt(data?.impact_equivalents?.trees ?? 0),
            sub: 'trees planted',
            color: '#6B8F5E',
            bg: '#D4E6CF',
        },
        {
            icon: '🚗',
            label: 'Cars Off the Road',
            value: fmt(data?.impact_equivalents?.cars_off_road ?? 0),
            sub: 'car-years avoided',
            color: '#D4900A',
            bg: '#FFF3CC',
        },
        {
            icon: '🏠',
            label: 'Homes Powered',
            value: fmt(data?.impact_equivalents?.homes_powered ?? 0),
            sub: 'Indian homes / month',
            color: '#D4900A',
            bg: '#FFF8E7',
        },
    ];

    return (
        <div ref={wrapRef} style={{ height: `${cards.length * 80}vh` }} className="relative">
            <div className="sticky top-0 h-screen overflow-hidden flex flex-col justify-center">
                {/* Label */}
                <p className="text-center mb-6 text-sm font-medium tracking-widest uppercase"
                    style={{ color: '#D4900A', fontFamily: 'Montserrat, sans-serif' }}>
                    ← Scroll to explore your impact →
                </p>
                {/* Track */}
                <div ref={trackRef} className="flex gap-6 px-16"
                    style={{ transition: 'transform 0.05s linear', willChange: 'transform' }}>
                    {cards.map((c, i) => (
                        <div key={i} className="flex-shrink-0 w-72 rounded-2xl p-8 flex flex-col gap-3"
                            style={{ background: c.bg, border: `2px solid ${c.color}22`, boxShadow: `0 8px 40px ${c.color}22` }}>
                            <span className="text-5xl">{c.icon}</span>
                            <p className="text-xs font-semibold tracking-wider uppercase"
                                style={{ color: c.color, fontFamily: 'Montserrat, sans-serif' }}>{c.label}</p>
                            <p className="text-3xl font-bold"
                                style={{ color: '#2C2C2C', fontFamily: 'Bebas Neue, sans-serif', letterSpacing: '0.04em' }}>
                                {c.value}
                            </p>
                            <p className="text-xs" style={{ color: '#6B6560' }}>{c.sub}</p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function ImpactPage() {
    const { data, isLoading, isError } = useQuery({
        queryKey: ['green-analytics'],
        queryFn: () => adminApi.getGreenAnalytics(),
        staleTime: 5 * 60_000,
    });

    // Hero animation
    const heroRef = useRef<HTMLDivElement>(null);
    const headlineRef = useRef<HTMLHeadingElement>(null);
    const subRef = useRef<HTMLParagraphElement>(null);

    useEffect(() => {
        if (!heroRef.current) return;
        anime.timeline({ easing: 'easeOutExpo' })
            .add({ targets: headlineRef.current, opacity: [0, 1], translateY: [60, 0], duration: 900 })
            .add({ targets: subRef.current, opacity: [0, 1], translateY: [30, 0], duration: 700 }, '-=400');
    }, []);

    // Floating particles on hero
    const particlesRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (!particlesRef.current) return;
        const dots = particlesRef.current.querySelectorAll('.particle');
        anime({
            targets: dots,
            translateX: () => anime.random(-40, 40),
            translateY: () => anime.random(-60, 60),
            scale: () => anime.random(0.5, 1.5),
            opacity: [0.3, 0.7],
            duration: () => anime.random(2000, 4000),
            loop: true,
            direction: 'alternate',
            delay: anime.stagger(150, { from: 'center' }),
            easing: 'easeInOutSine',
        });
    }, []);

    // CO2 counter section
    const co2Section = useInView(0.3);
    const co2Val = useCountUp(data?.co2_avoided_tonnes ?? 0, co2Section.inView);
    const creditsVal = useCountUp(data?.carbon_credits ?? 0, co2Section.inView);
    const creditUsdVal = useCountUp(data?.credit_value_usd ?? 0, co2Section.inView);
    const energyVal = useCountUp(data?.energy_kwh_30d ?? 0, co2Section.inView);

    // Revenue section
    const revSection = useInView(0.3);
    const revVal = useCountUp(data?.revenue_lost_inr ?? 0, revSection.inView);

    // Revenue section animation
    const revCardsRef = useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (!revSection.inView || !revCardsRef.current) return;
        anime({
            targets: revCardsRef.current.querySelectorAll('.fault-row'),
            translateX: [-40, 0],
            opacity: [0, 1],
            delay: anime.stagger(100),
            duration: 500,
            easing: 'easeOutQuart',
        });
    }, [revSection.inView]);

    // Pin + parallax scroll for hero
    useEffect(() => {
        const onScroll = () => {
            if (!heroRef.current) return;
            const y = window.scrollY;
            heroRef.current.style.backgroundPositionY = `${y * 0.4}px`;
        };
        window.addEventListener('scroll', onScroll, { passive: true });
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    if (isLoading) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <Loader2 className="h-10 w-10 animate-spin" style={{ color: '#D4900A' }} />
            </div>
        );
    }

    if (isError) {
        return (
            <div className="flex items-center justify-center h-[60vh]">
                <p className="text-sm" style={{ color: '#C0392B' }}>
                    Failed to load green analytics. Make sure the server is running.
                </p>
            </div>
        );
    }

    const pr = data?.pr_percent ?? 0;
    const cuf = data?.cuf_percent ?? 0;
    const prBenchmark = pr >= 80 ? 'Good ≥80%' : pr >= 70 ? 'Average 70–80%' : 'Below Par <70%';
    const prColor = pr >= 80 ? '#6B8F5E' : pr >= 70 ? '#E67E22' : '#C0392B';
    const cufBenchmark = cuf >= 18 ? 'Good ≥18%' : cuf >= 14 ? 'Average' : 'Improving';
    const cufColor = cuf >= 18 ? '#6B8F5E' : cuf >= 14 ? '#E67E22' : '#D4900A';

    return (
        <div style={{ background: '#FFFDF5', fontFamily: 'Inter, sans-serif', color: '#2C2C2C' }}>

            {/* ── HERO ──────────────────────────────────────────── */}
            <div ref={heroRef} className="relative overflow-hidden flex flex-col items-center justify-center text-center"
                style={{
                    minHeight: '92vh',
                    background: 'linear-gradient(160deg, #D4900A 0%, #F5A623 45%, #FFD166 100%)',
                }}>

                {/* floating particles */}
                <div ref={particlesRef} className="absolute inset-0 pointer-events-none">
                    {Array.from({ length: 18 }).map((_, i) => (
                        <div key={i} className="particle absolute rounded-full"
                            style={{
                                width: `${6 + (i % 5) * 4}px`, height: `${6 + (i % 5) * 4}px`,
                                background: '#FFF8E7', opacity: 0.4,
                                top: `${10 + (i * 13) % 80}%`, left: `${5 + (i * 17) % 90}%`,
                            }} />
                    ))}
                </div>

                {/* Sun graphic */}
                <div className="w-52 h-52 mb-6 relative z-10">
                    <SunSVG />
                </div>

                <h1 ref={headlineRef} className="relative z-10 px-4"
                    style={{
                        fontFamily: 'Bebas Neue, sans-serif',
                        fontSize: 'clamp(3rem, 8vw, 7rem)',
                        color: '#FFF8E7',
                        letterSpacing: '0.04em',
                        lineHeight: 1.05,
                        textShadow: '0 4px 24px rgba(0,0,0,0.18)',
                        opacity: 0,
                    }}>
                    Every Watt Tells<br />a Story
                </h1>
                <p ref={subRef} className="relative z-10 max-w-xl px-6 mt-4 text-lg"
                    style={{ color: '#FFF8E7CC', fontFamily: 'Montserrat, sans-serif', fontWeight: 600, opacity: 0 }}>
                    Discover the real-world environmental and economic impact your solar fleet has made this month.
                </p>

                {/* Scroll cue */}
                <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 z-10">
                    <p className="text-xs uppercase tracking-widest" style={{ color: '#FFF8E7AA', fontFamily: 'Montserrat, sans-serif' }}>
                        Scroll to discover
                    </p>
                    <svg width="24" height="40" viewBox="0 0 24 40" className="animate-bounce">
                        <rect x="4" y="2" width="16" height="28" rx="8" fill="none" stroke="#FFF8E799" strokeWidth="2" />
                        <rect x="10" y="6" width="4" height="8" rx="2" fill="#FFF8E799" />
                        <polyline points="6,34 12,40 18,34" fill="none" stroke="#FFF8E799" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                </div>
            </div>

            {/* ── CO2 COUNTER SECTION ──────────────────────────── */}
            <div ref={co2Section.ref} className="py-24 px-4"
                style={{ background: '#FFF8E7' }}>
                {/* Watermark */}
                <div className="text-center mb-2 relative">
                    <span className="absolute inset-0 flex items-center justify-center pointer-events-none select-none"
                        style={{
                            fontFamily: 'Bebas Neue, sans-serif', fontSize: 'clamp(5rem, 15vw, 12rem)',
                            color: '#FFE599', opacity: 0.18, letterSpacing: '0.06em', zIndex: 0,
                        }}>
                        IMPACT
                    </span>
                    <p className="relative z-10 text-xs uppercase tracking-widest mb-2"
                        style={{ color: '#D4900A', fontFamily: 'Montserrat, sans-serif', fontWeight: 700 }}>
                        Environmental Contribution · Last 30 Days
                    </p>
                    <h2 className="relative z-10 text-4xl font-bold"
                        style={{ fontFamily: 'Bebas Neue, sans-serif', color: '#2C2C2C', letterSpacing: '0.04em', fontSize: 'clamp(2rem, 5vw, 4rem)' }}>
                        Your Green Footprint
                    </h2>
                </div>

                <div className="max-w-5xl mx-auto mt-12 grid grid-cols-2 md:grid-cols-4 gap-6">
                    {[
                        { label: 'Energy Generated', unit: 'kWh', val: energyVal, icon: '⚡', color: '#D4900A' },
                        { label: 'CO₂ Avoided', unit: 'tonnes', val: co2Val, icon: '🌿', color: '#6B8F5E' },
                        { label: 'Carbon Credits', unit: 'credits', val: creditsVal, icon: '🏅', color: '#D4900A' },
                        { label: 'Market Value', unit: 'USD', val: creditUsdVal, icon: '💵', color: '#6B8F5E' },
                    ].map((s, i) => (
                        <div key={i} className="rounded-2xl p-6 flex flex-col items-center gap-2 text-center transition-transform hover:-translate-y-1"
                            style={{ background: '#FFFDF5', border: '2px solid #E8D5A3', boxShadow: '0 4px 20px #D4900A11' }}>
                            <span className="text-4xl">{s.icon}</span>
                            <p className="text-xs uppercase tracking-wider font-semibold"
                                style={{ color: s.color, fontFamily: 'Montserrat, sans-serif' }}>{s.label}</p>
                            <p className="font-bold" style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: '2.4rem', color: '#2C2C2C', letterSpacing: '0.04em' }}>
                                {s.label === 'Energy Generated' ? fmt(s.val) : s.label === 'Market Value' ? `$${fmt(s.val)}` : fmtDec(s.val)}
                            </p>
                            <p className="text-xs" style={{ color: '#6B6560' }}>{s.unit}</p>
                        </div>
                    ))}
                </div>

                {/* Leaf decorations */}
                <div className="flex justify-center gap-4 mt-12">
                    {[false, true, false].map((a, i) => <LeafSVG key={i} animated={a} />)}
                </div>
            </div>

            {/* ── HORIZONTAL SCROLL STORY ──────────────────────── */}
            <div style={{ background: '#FFFDF5' }}>
                <div className="text-center pt-16 pb-4">
                    <p className="text-xs uppercase tracking-widest mb-2"
                        style={{ color: '#D4900A', fontFamily: 'Montserrat, sans-serif', fontWeight: 700 }}>
                        The Full Picture
                    </p>
                    <h2 style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 'clamp(2rem, 5vw, 4rem)', letterSpacing: '0.04em' }}>
                        Scroll Through Your Impact
                    </h2>
                </div>
                <HorizontalStory data={data} />
            </div>

            {/* ── REVENUE LOSS SECTION ─────────────────────────── */}
            <div ref={revSection.ref} className="py-24 px-4"
                style={{ background: 'linear-gradient(180deg, #FFF3CC 0%, #FFF8E7 100%)' }}>
                <div className="max-w-4xl mx-auto">
                    <div className="text-center mb-10">
                        <p className="text-xs uppercase tracking-widest mb-2"
                            style={{ color: '#C0392B', fontFamily: 'Montserrat, sans-serif', fontWeight: 700 }}>
                            Economic Intelligence
                        </p>
                        <h2 style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 'clamp(2rem, 5vw, 4rem)', letterSpacing: '0.04em' }}>
                            Revenue Lost to Faults
                        </h2>
                        <p className="text-sm mt-2" style={{ color: '#6B6560' }}>
                            Faults cost real money. Here's the financial case for predictive maintenance.
                        </p>
                    </div>

                    {/* Big number */}
                    <div className="text-center mb-10 p-8 rounded-2xl"
                        style={{ background: '#FFF8E7', border: '3px solid #C0392B22', boxShadow: '0 8px 40px #C0392B11' }}>
                        <p className="text-xs uppercase tracking-wider mb-2 font-semibold" style={{ color: '#C0392B', fontFamily: 'Montserrat, sans-serif' }}>
                            Total Revenue Lost (Active Faults)
                        </p>
                        <p style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 'clamp(3rem, 10vw, 6rem)', color: '#C0392B', letterSpacing: '0.04em' }}>
                            ₹{fmt(revVal)}
                        </p>
                        <p className="text-xs mt-1" style={{ color: '#6B6560' }}>
                            Based on ₹4/kWh solar PPA tariff · Inverters in Category D/E
                        </p>
                    </div>

                    {/* Breakdown */}
                    {(data?.revenue_lost_breakdown?.length ?? 0) > 0 ? (
                        <div ref={revCardsRef} className="space-y-3">
                            <p className="text-sm font-semibold mb-2" style={{ fontFamily: 'Montserrat, sans-serif', color: '#2C2C2C' }}>
                                Affected Inverters
                            </p>
                            {(data?.revenue_lost_breakdown ?? []).map((row: any, i: number) => (
                                <div key={i} className="fault-row flex items-center gap-4 p-4 rounded-xl"
                                    style={{ background: '#FFF8E7', border: '1.5px solid #E8D5A3', boxShadow: '0 2px 12px #D4900A11' }}>
                                    <div className="h-10 w-10 rounded-full flex items-center justify-center flex-shrink-0"
                                        style={{ background: '#C0392B15' }}>
                                        <span className="text-lg">⚠️</span>
                                    </div>
                                    <div className="flex-1">
                                        <p className="font-semibold text-sm" style={{ fontFamily: 'Montserrat, sans-serif' }}>{row.inverter}</p>
                                        <p className="text-xs" style={{ color: '#6B6560' }}>{row.hours}h in fault state</p>
                                    </div>
                                    <p className="font-bold text-lg" style={{ color: '#C0392B', fontFamily: 'Bebas Neue, sans-serif', letterSpacing: '0.04em' }}>
                                        ₹{fmt(row.loss_inr)}
                                    </p>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="text-center p-8 rounded-2xl"
                            style={{ background: '#D4E6CF50', border: '2px solid #6B8F5E33' }}>
                            <span className="text-5xl">✅</span>
                            <p className="mt-4 font-semibold" style={{ color: '#6B8F5E', fontFamily: 'Montserrat, sans-serif' }}>
                                No active fault losses! All inverters are performing within normal range.
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* ── PR / CUF GAUGES ──────────────────────────────── */}
            <div className="py-24 px-4 text-center" style={{ background: '#FFF8E7' }}>
                <div className="mb-12">
                    <p className="text-xs uppercase tracking-widest mb-2"
                        style={{ color: '#D4900A', fontFamily: 'Montserrat, sans-serif', fontWeight: 700 }}>
                        Performance KPIs
                    </p>
                    <h2 style={{ fontFamily: 'Bebas Neue, sans-serif', fontSize: 'clamp(2rem, 5vw, 4rem)', letterSpacing: '0.04em' }}>
                        Industry Benchmarks
                    </h2>
                    <p className="text-sm mt-2 max-w-lg mx-auto" style={{ color: '#6B6560' }}>
                        Performance Ratio and Capacity Utilization Factor — the two KPIs lenders, investors, and regulators watch.
                    </p>
                </div>

                <div className="flex flex-wrap items-center justify-center gap-16 max-w-3xl mx-auto">
                    <div className="flex flex-col items-center gap-6">
                        <GaugeSVG percent={pr} label="Performance Ratio (PR)" color={prColor} benchmarkLabel={prBenchmark} />
                        <p className="text-xs max-w-xs" style={{ color: '#6B6560' }}>
                            Ratio of actual energy output vs. theoretical maximum. Industry target: ≥80%.
                        </p>
                    </div>
                    <div className="flex flex-col items-center gap-6">
                        <GaugeSVG percent={cuf} label="Capacity Utilization Factor (CUF)" color={cufColor} benchmarkLabel={cufBenchmark} />
                        <p className="text-xs max-w-xs" style={{ color: '#6B6560' }}>
                            Annualized yield as % of total installed capacity. Typical Indian solar: 18–22%.
                        </p>
                    </div>
                </div>
            </div>

            {/* ── CLOSING CTA ──────────────────────────────────── */}
            <div className="py-28 px-4 text-center"
                style={{ background: 'linear-gradient(160deg, #2C1A0E 0%, #D4900A 100%)' }}>
                <div className="w-24 h-24 mx-auto mb-6">
                    <SunSVG />
                </div>
                <h2 className="text-5xl md:text-7xl font-bold mb-4"
                    style={{ fontFamily: 'Bebas Neue, sans-serif', color: '#FFE599', letterSpacing: '0.06em' }}>
                    The Future is Solar
                </h2>
                <p className="max-w-xl mx-auto text-lg mb-8"
                    style={{ color: '#FFF8E7AA', fontFamily: 'Montserrat, sans-serif', fontWeight: 600 }}>
                    Every fault detected early is a tree saved. Every kWh optimized is a step closer to net zero.
                    Lumin AI is your partner in a cleaner, more profitable future.
                </p>
                <div className="inline-flex gap-3 flex-wrap justify-center">
                    <div className="px-6 py-3 rounded-full text-sm font-bold"
                        style={{ background: '#FFD166', color: '#2C2C2C', fontFamily: 'Montserrat, sans-serif' }}>
                        🌳 {fmt(data?.impact_equivalents?.trees ?? 0)} trees equivalent
                    </div>
                    <div className="px-6 py-3 rounded-full text-sm font-bold"
                        style={{ background: '#FFD16622', color: '#FFD166', border: '2px solid #FFD16644', fontFamily: 'Montserrat, sans-serif' }}>
                        🚗 {fmt(data?.impact_equivalents?.cars_off_road ?? 0)} car-years avoided
                    </div>
                    <div className="px-6 py-3 rounded-full text-sm font-bold"
                        style={{ background: '#FFD16622', color: '#FFD166', border: '2px solid #FFD16644', fontFamily: 'Montserrat, sans-serif' }}>
                        🏠 {fmt(data?.impact_equivalents?.homes_powered ?? 0)} homes powered
                    </div>
                </div>
            </div>

        </div>
    );
}
