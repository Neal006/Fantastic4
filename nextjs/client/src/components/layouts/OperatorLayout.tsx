import { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from 'next-themes';
import { useQuery } from '@tanstack/react-query';
import { operatorApi } from '@/lib/api';
import {
  LayoutDashboard, AlertTriangle, MessageCircle, User, LogOut,
  Sun, Moon, Bell, Menu, X, ChevronDown, ChevronRight, Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export default function OperatorLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { theme, setTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedPlants, setExpandedPlants] = useState<string[]>([]);

  if (!user) return null;

  const { data: plants = [] } = useQuery({
    queryKey: ['operator-plants'],
    queryFn: () => operatorApi.getPlants(),
    staleTime: 60000,
  });

  const { data: alerts = [] } = useQuery({
    queryKey: ['operator-alerts'],
    queryFn: () => operatorApi.getAlerts(),
    refetchInterval: 30000,
  });

  const pendingAlerts = (alerts as any[]).filter((a: any) => !a.acknowledged);
  const criticalPending = pendingAlerts.filter((a: any) => a.type === 'critical');

  const togglePlant = (id: string) => {
    setExpandedPlants(prev => prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]);
  };

  const isActive = (path: string) => location.pathname === path;

  const navItem = (path: string, icon: React.ReactNode, label: string, badge?: number) => (
    <button
      onClick={() => { navigate(path); if (window.innerWidth < 1024) setSidebarOpen(false); }}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${isActive(path)
        ? 'bg-primary/20 text-primary-foreground'
        : 'text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground'
        }`}
    >
      {icon}
      <span className="flex-1 text-left">{label}</span>
      {badge !== undefined && badge > 0 && (
        <Badge variant="destructive" className="text-xs h-5 min-w-[20px] flex items-center justify-center">{badge}</Badge>
      )}
    </button>
  );

  const initials = user.name.split(' ').map(n => n[0]).join('');

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <aside className={`
        fixed lg:relative z-40 h-full transition-transform duration-300
        ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0 lg:w-0 lg:overflow-hidden'}
        w-[260px] flex-shrink-0 sw-sidebar flex flex-col
      `}>
        <div className="p-4 flex items-center gap-2 border-b border-sidebar-border">
          <div className="h-8 w-8 rounded-lg bg-primary flex items-center justify-center">
            <Zap className="h-5 w-5 text-primary-foreground" />
          </div>
          <span className="font-bold text-lg text-sidebar-foreground">Lumin AI</span>
        </div>

        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {navItem('/operator', <LayoutDashboard className="h-4 w-4" />, 'Dashboard')}

          <div className="pt-2 pb-1 px-3 text-xs uppercase tracking-wider text-sidebar-foreground/40 font-semibold">
            Plants
          </div>
          {(plants as any[]).map((plant: any) => (
            <div key={plant.id}>
              <button
                onClick={() => togglePlant(plant.id)}
                className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-sidebar-foreground/70 hover:bg-sidebar-accent transition-colors"
              >
                {expandedPlants.includes(plant.id) ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
                <span className="flex-1 text-left font-medium">{plant.name}</span>
              </button>
              {expandedPlants.includes(plant.id) && (
                <div className="ml-6 space-y-0.5">
                  {(plant.blocks || []).map((block: any) => (
                    <button
                      key={block.id}
                      onClick={() => { navigate(`/operator/plant/${plant.id}/block/${block.id}`); if (window.innerWidth < 1024) setSidebarOpen(false); }}
                      className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors ${location.pathname === `/operator/plant/${plant.id}/block/${block.id}`
                        ? 'bg-primary/20 text-primary-foreground font-medium'
                        : 'text-sidebar-foreground/50 hover:text-sidebar-foreground/80'
                        }`}
                    >
                      {block.name} ({block.inverterCount ?? '—'})
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}

          <div className="pt-3" />
          {navItem('/operator/alerts', <AlertTriangle className="h-4 w-4" />, 'Alerts', pendingAlerts.length)}
          {navItem('/operator/chatbot', <MessageCircle className="h-4 w-4" />, 'AI Chatbot')}
          {navItem('/operator/profile', <User className="h-4 w-4" />, 'Profile')}
        </nav>

        <div className="p-3 border-t border-sidebar-border">
          <button
            onClick={() => { logout(); navigate('/login'); }}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-sidebar-foreground/70 hover:bg-sidebar-accent transition-colors"
          >
            <LogOut className="h-4 w-4" />
            Logout
          </button>
        </div>
      </aside>

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-30 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Bar */}
        <header className="h-14 border-b flex items-center gap-3 px-4 bg-card flex-shrink-0">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>

          <div className="flex-1" />

          {criticalPending.length > 0 && (
            <Badge variant="destructive" className="text-xs animate-pulse">
              {criticalPending.length} Critical
            </Badge>
          )}

          <Button variant="ghost" size="icon" className="relative" onClick={() => navigate('/operator/alerts')}>
            <Bell className="h-5 w-5" />
            {pendingAlerts.length > 0 && (
              <span className="absolute -top-0.5 -right-0.5 h-4 w-4 rounded-full bg-destructive text-[10px] text-destructive-foreground flex items-center justify-center font-bold">
                {pendingAlerts.length}
              </span>
            )}
          </Button>

          <Button variant="ghost" size="icon" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>
            {theme === 'dark' ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </Button>

          <div className="flex items-center gap-2 ml-1">
            <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center text-xs font-bold text-primary-foreground">
              {initials}
            </div>
            <div className="hidden sm:block text-right">
              <p className="text-sm font-medium leading-none">{user.name}</p>
              <p className="text-xs text-muted-foreground capitalize">{user.role}</p>
            </div>
          </div>
        </header>

        {/* Critical alert banner */}
        {criticalPending.length > 0 && (
          <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-2 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-destructive" />
            <span className="text-sm font-medium text-destructive">
              {criticalPending.length} unacknowledged critical alert{criticalPending.length > 1 ? 's' : ''} require attention
            </span>
            <Button variant="outline" size="sm" className="ml-auto h-7 text-xs" onClick={() => navigate('/operator/alerts')}>
              View Alerts
            </Button>
          </div>
        )}

        <main className="flex-1 overflow-y-auto p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
