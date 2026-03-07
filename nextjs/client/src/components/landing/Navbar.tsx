import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { MagneticButton } from '@/components/ui/Motion';

const navLinks = ['Platform', 'Technology', 'Impact', 'Research'];

export const Navbar = () => {
    const [scrolled, setScrolled] = useState(false);
    const [hidden, setHidden] = useState(false);
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        let lastScrollY = window.scrollY;
        const handleScroll = () => {
            const currentScrollY = window.scrollY;
            setScrolled(currentScrollY > 80);

            if (currentScrollY > lastScrollY && currentScrollY > 80) {
                setHidden(true); // scrolling down
            } else {
                setHidden(false); // scrolling up
            }
            lastScrollY = currentScrollY;
        };
        window.addEventListener('scroll', handleScroll, { passive: true });
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <>
            <motion.nav
                initial={{ y: -100 }}
                animate={{ y: hidden ? -100 : 0 }}
                transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
                style={{
                    position: 'fixed',
                    top: 0,
                    left: 0,
                    right: 0,
                    height: '80px',
                    zIndex: 9990,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0 5%',
                    background: scrolled ? 'rgba(255, 253, 245, 0.8)' : 'transparent',
                    backdropFilter: scrolled ? 'blur(12px)' : 'none',
                    borderBottom: scrolled ? '1px solid var(--color-brand-light)' : '1px solid transparent',
                    boxShadow: scrolled ? 'var(--shadow-card)' : 'none',
                    transition: 'background 0.3s, backdrop-filter 0.3s, border-bottom 0.3s, box-shadow 0.3s',
                }}
            >
                {/* Logo */}
                <div style={{
                    display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer'
                }} onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
                    <span style={{ fontSize: '24px', color: 'var(--color-brand-accent)' }}>⚡</span>
                    <span style={{
                        fontFamily: 'var(--font-heading)',
                        fontWeight: 700,
                        fontSize: '20px',
                        color: 'var(--color-text-primary)',
                        letterSpacing: '-0.02em',
                    }}>
                        Lumin AI
                    </span>
                </div>

                {/* Desktop Links */}
                <div className="hidden md:flex" style={{ alignItems: 'center', gap: '32px' }}>
                    {navLinks.map((link) => (
                        <a key={link} href={`#${link.toLowerCase()}`} className="nav-link" style={{
                            fontFamily: 'var(--font-body)',
                            fontWeight: 500,
                            fontSize: 'var(--text-small)',
                            color: 'var(--color-text-primary)'
                        }}>
                            {link}
                        </a>
                    ))}

                    <MagneticButton
                        onClick={() => navigate('/login')}
                        style={{
                            padding: '10px 24px',
                            borderRadius: 'var(--radius-sm)',
                            background: 'var(--color-brand-primary)',
                            color: 'var(--color-text-inverse)',
                            fontFamily: 'var(--font-body)',
                            fontWeight: 600,
                            fontSize: 'var(--text-small)',
                            border: 'none',
                            cursor: 'pointer',
                        }}
                    >
                        Get Started
                    </MagneticButton>
                </div>

                {/* Mobile Hamburger toggle */}
                <button
                    className="md:hidden"
                    onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                    style={{
                        background: 'none', border: 'none',
                        fontSize: '24px', color: 'var(--color-text-primary)',
                        cursor: 'pointer',
                    }}
                >
                    {mobileMenuOpen ? '✕' : '☰'}
                </button>
            </motion.nav>

            {/* Mobile Menu Drawer */}
            <AnimatePresence>
                {mobileMenuOpen && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.4, ease: [0.25, 0.46, 0.45, 0.94] }}
                        style={{
                            position: 'fixed',
                            top: '80px',
                            left: 0,
                            right: 0,
                            background: 'var(--color-bg-secondary)',
                            zIndex: 9989,
                            overflow: 'hidden',
                            borderBottom: '1px solid var(--color-border)',
                            display: 'flex',
                            flexDirection: 'column',
                            padding: '24px 5%',
                            gap: '20px',
                            boxShadow: 'var(--shadow-card)',
                        }}
                        className="md:hidden"
                    >
                        {navLinks.map((link, i) => (
                            <motion.a
                                key={link}
                                href={`#${link.toLowerCase()}`}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: 0.1 + i * 0.05 }}
                                onClick={() => setMobileMenuOpen(false)}
                                style={{
                                    fontFamily: 'var(--font-heading)',
                                    fontWeight: 600,
                                    fontSize: 'var(--text-h3)',
                                    color: 'var(--color-text-primary)',
                                    textDecoration: 'none',
                                }}
                            >
                                {link}
                            </motion.a>
                        ))}
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 + navLinks.length * 0.05 }}
                            style={{ marginTop: '16px' }}
                        >
                            <button
                                onClick={() => navigate('/login')}
                                style={{
                                    width: '100%',
                                    padding: '16px 24px',
                                    borderRadius: 'var(--radius-sm)',
                                    background: 'var(--color-brand-primary)',
                                    color: 'var(--color-text-inverse)',
                                    fontFamily: 'var(--font-body)',
                                    fontWeight: 600,
                                    fontSize: '1rem',
                                    border: 'none',
                                    cursor: 'pointer',
                                }}
                            >
                                Get Started
                            </button>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>

            <style>{`
        .nav-link {
          position: relative;
          text-decoration: none;
        }
        .nav-link::after {
          content: '';
          position: absolute;
          width: 0;
          height: 1.5px;
          bottom: -4px;
          left: 0;
          background-color: var(--color-brand-primary);
          transition: width 0.3s ease;
        }
        .nav-link:hover::after {
          width: 100%;
        }
      `}</style>
        </>
    );
};
