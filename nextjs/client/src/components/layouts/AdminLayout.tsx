import { useState } from 'react';
import { Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/contexts/AuthContext';
import { useTheme } from 'next-themes';
import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import {
  LayoutDashboard, Building2, Users, Monitor, AlertTriangle,
  FileText, Settings, User, LogOut, Sun, Moon, Bell, Menu, X, Zap, Leaf
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export default function AdminLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const { theme, setTheme } = useTheme();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  if (!user) return null;

  const { data: alerts = [] } = useQuery({
    queryKey: ['admin-alerts-nav'],
    queryFn: () => adminApi.getAlerts({ acknowledged: 'false' }),
    refetchInterval: 30000,
  });

  const pendingCount = (alerts as any[]).length;
  const isActive = (path: string) => location.pathname === path || location.pathname.startsWith(path + '/');

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
          <Badge className="ml-auto text-[10px] bg-primary/20 text-primary-foreground border-0">Admin</Badge>
        </div>

        <nav className="flex-1 p-3 space-y-1 overflow-y-auto">
          {navItem('/admin', <LayoutDashboard className="h-4 w-4" />, 'Dashboard')}
          {navItem('/admin/plants', <Building2 className="h-4 w-4" />, 'Plant Management')}
          {navItem('/admin/operators', <Users className="h-4 w-4" />, 'Operator Management')}
          {navItem('/admin/monitoring', <Monitor className="h-4 w-4" />, 'Live Monitoring')}
          {navItem('/admin/alerts', <AlertTriangle className="h-4 w-4" />, 'Alerts', pendingCount)}
          {navItem('/admin/audit-logs', <FileText className="h-4 w-4" />, 'Audit Logs')}
          {navItem('/admin/settings', <Settings className="h-4 w-4" />, 'Settings')}
          {navItem('/admin/profile', <User className="h-4 w-4" />, 'Profile')}
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

      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-30 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      <div className="flex-1 flex flex-col min-w-0">
        <header className="h-14 border-b flex items-center gap-3 px-4 bg-card flex-shrink-0">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setSidebarOpen(!sidebarOpen)}>
            {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
          <div className="flex-1" />

          <Button variant="ghost" size="icon" className="relative" onClick={() => navigate('/admin/alerts')}>
            <Bell className="h-5 w-5" />
            {pendingCount > 0 && (
              <span className="absolute -top-0.5 -right-0.5 h-4 w-4 rounded-full bg-destructive text-[10px] text-destructive-foreground flex items-center justify-center font-bold">
                {pendingCount}
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

        <main className="flex-1 overflow-y-auto p-4 lg:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
