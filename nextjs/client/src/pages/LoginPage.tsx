import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { motion } from 'framer-motion';
import { MagneticButton } from '@/components/ui/Motion';
import { BgPolygon, RotatingDiamond } from '@/components/ui/Shapes';

export default function LoginPage() {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [shake, setShake] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    const ok = await login(userId, password);
    setLoading(false);
    if (ok) {
      const user = JSON.parse(sessionStorage.getItem('sw-user') || '{}');
      navigate(user.role === 'admin' ? '/admin' : '/operator');
    } else {
      setError('Invalid credentials. Please try again.');
      setShake(true);
      setTimeout(() => setShake(false), 500);
    }
  };

  const inputStyle = {
    width: '100%',
    background: 'rgba(255, 253, 245, 0.07)',
    border: '1px solid rgba(255, 214, 102, 0.2)',
    borderRadius: 'var(--radius-sm)',
    color: 'var(--color-text-inverse)',
    padding: '14px 16px',
    fontFamily: 'var(--font-body)',
    fontSize: 'var(--text-body)',
    outline: 'none',
    transition: 'border-color 0.2s ease, background 0.2s ease',
    marginBottom: '12px',
  };

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden"
      style={{ background: 'radial-gradient(ellipse 80% 70% at 50% 40%, rgba(212,144,10,0.35) 0%, #1a1208 100%)' }}>

      {/* Background Shapes */}
      <BgPolygon style={{ top: '-10%', left: '-10%', transform: 'rotate(15deg) scale(1.5)', opacity: 0.2, filter: 'blur(8px)' }} />
      <BgPolygon style={{ bottom: '-10%', right: '-10%', transform: 'rotate(195deg) scale(0.8)', opacity: 0.1, filter: 'blur(4px)' }} />

      {/* Floating Particles */}
      {Array.from({ length: 20 }).map((_, i) => (
        <motion.div
          key={i}
          initial={{ y: 0, opacity: 0.3 }}
          animate={{ y: [0, -100], opacity: [0.3, 0] }}
          transition={{ duration: 3 + Math.random() * 4, repeat: Infinity, delay: Math.random() * 4, ease: "linear" }}
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

      {/* Slowly rotating large diamond outline */}
      <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', opacity: 0.04, pointerEvents: 'none' }}>
        <RotatingDiamond size={600} speed={40} />
      </div>

      {/* Small accent diamonds */}
      <RotatingDiamond size={40} speed={20} style={{ position: 'absolute', top: '20%', left: '15%', opacity: 0.3 }} />
      <RotatingDiamond size={24} speed={15} style={{ position: 'absolute', bottom: '25%', right: '20%', opacity: 0.2 }} />

      {/* Glass Panel */}
      <motion.div
        initial={{ opacity: 0, scale: 0.92, y: 24 }}
        animate={{ opacity: 1, scale: 1, y: 0, x: shake ? [-4, 4, -4, 4, 0] : 0 }}
        transition={{
          duration: 0.7,
          ease: [0.34, 1.56, 0.64, 1],
          x: { duration: 0.4 } // Shake duration
        }}
        className="relative z-10 w-full max-w-[420px] mx-auto px-6 md:px-0"
      >
        <div style={{
          background: 'rgba(255, 253, 245, 0.10)',
          backdropFilter: 'blur(24px) saturate(180%)',
          border: '1px solid rgba(255, 214, 102, 0.3)',
          borderRadius: '20px',
          boxShadow: '0 24px 80px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08)',
          padding: '48px 40px',
        }}>
          {/* Header */}
          <div className="flex flex-col items-center mb-6 text-center">
            <span style={{ fontSize: '52px', marginBottom: '8px', lineHeight: 1 }}>⚡</span>
            <h1 style={{ fontFamily: 'var(--font-display)', fontSize: '3rem', color: 'var(--color-brand-light)', letterSpacing: '0.05em', lineHeight: 1 }}>
              LUMIN AI
            </h1>
            <p style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: '0.8rem', color: 'var(--color-text-inverse)', opacity: 0.7, letterSpacing: '0.15em', textTransform: 'uppercase', marginTop: '4px' }}>
              Intelligence Platform
            </p>
          </div>

          <div style={{ borderBottom: '1px solid rgba(255, 214, 102, 0.15)', margin: '24px 0' }} />

          {/* Form */}
          <form onSubmit={handleSubmit}>
            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <input
                type="text"
                placeholder="User ID (e.g. admin@lumin.ai)"
                value={userId}
                onChange={e => setUserId(e.target.value)}
                required
                style={inputStyle}
                className="focus:border-[var(--color-brand-light)] focus:bg-[rgba(255,253,245,0.12)] placeholder:text-[rgba(255,248,231,0.4)]"
              />
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.38 }}>
              <input
                type="password"
                placeholder="Password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                style={inputStyle}
                className="focus:border-[var(--color-brand-light)] focus:bg-[rgba(255,253,245,0.12)] placeholder:text-[rgba(255,248,231,0.4)]"
              />
            </motion.div>

            {error && (
              <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-sm font-medium mb-3 mt-1" style={{ color: 'var(--color-error)' }}>
                {error}
              </motion.p>
            )}

            <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.46 }}>
              <MagneticButton
                type="submit"
                disabled={loading}
                style={{
                  width: '100%',
                  marginTop: '12px',
                  padding: '16px 0',
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--color-brand-primary)',
                  color: 'var(--color-text-inverse)',
                  fontFamily: 'var(--font-display)',
                  fontSize: '1.4rem',
                  border: 'none',
                  cursor: loading ? 'not-allowed' : 'pointer',
                  opacity: loading ? 0.7 : 1,
                  letterSpacing: '0.05em'
                }}
              >
                {loading ? 'AUTHENTICATING...' : 'SIGN IN'}
              </MagneticButton>
            </motion.div>
          </form>

          {/* Footer */}
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }} className="mt-8 text-center pt-2">
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '0.85rem', color: 'var(--color-text-inverse)', opacity: 0.5 }}>
              Don't have access? <a href="#" style={{ color: 'var(--color-brand-light)' }}>Request Demo &rarr;</a>
            </p>
          </motion.div>
        </div>
      </motion.div>

      {/* Demo Credentials Helper */}
      <div className="absolute bottom-4 left-4 p-4 rounded-lg bg-[rgba(255,253,245,0.05)] backdrop-blur-sm border border-[rgba(255,214,102,0.1)] z-20 hidden md:block">
        <p className="text-xs font-semibold mb-2" style={{ color: 'var(--color-brand-light)', opacity: 0.8 }}>Demo Credentials</p>
        <div className="space-y-1 text-[11px]" style={{ color: 'var(--color-text-inverse)', opacity: 0.6 }}>
          <p><span className="font-mono text-white opacity-90 tracking-wider">admin@lumin.ai</span> / <span className="font-mono tracking-wider">Admin@123!</span></p>
          <p><span className="font-mono text-white opacity-90 tracking-wider">arjun.mehta@lumin.ai</span> / <span className="font-mono tracking-wider">Op@12345</span></p>
        </div>
      </div>
    </div>
  );
}
