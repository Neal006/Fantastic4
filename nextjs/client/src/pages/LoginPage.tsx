import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { Sun, Moon, Loader2, Zap } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { useTheme } from 'next-themes';

export default function LoginPage() {
  const [userId, setUserId] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [shake, setShake] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const { theme, setTheme } = useTheme();

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

  return (
    <div className="min-h-screen sw-login-gradient flex items-center justify-center p-4 relative">
      <button
        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        className="absolute top-4 right-4 p-2 rounded-lg bg-background/20 backdrop-blur-sm text-primary-foreground hover:bg-background/30 transition-colors"
      >
        {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
      </button>

      <Card className={`w-full max-w-md shadow-2xl border-0 ${shake ? 'animate-shake' : ''}`}>
        <CardHeader className="text-center pb-2 pt-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <div className="h-10 w-10 rounded-xl bg-primary flex items-center justify-center">
              <Zap className="h-6 w-6 text-primary-foreground" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">SolarWatch</h1>
          </div>
          <p className="text-sm text-muted-foreground">Solar Plant Inverter Monitoring System</p>
        </CardHeader>
        <CardContent className="px-8 pb-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="userId">User ID</Label>
              <Input
                id="userId"
                placeholder="Enter your user ID"
                value={userId}
                onChange={e => setUserId(e.target.value)}
                required
                autoFocus
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
              />
            </div>

            {error && (
              <p className="text-sm text-destructive font-medium">{error}</p>
            )}

            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Sign In
            </Button>
          </form>

          <div className="mt-6 p-4 rounded-lg bg-muted">
            <p className="text-xs font-semibold text-muted-foreground mb-2">Demo Credentials</p>
            <div className="space-y-1 text-xs text-muted-foreground">
              <p><span className="font-mono font-medium text-foreground">admin@solarwatch.in</span> / <span className="font-mono">Admin@123!</span> — Admin</p>
              <p><span className="font-mono font-medium text-foreground">arjun.mehta@solarwatch.in</span> / <span className="font-mono">Op@12345</span> — Operator</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
