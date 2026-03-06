import { useQuery } from '@tanstack/react-query';
import { useAuth } from '@/contexts/AuthContext';
import { operatorApi } from '@/lib/api';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';
import { Cpu, CheckCircle, AlertTriangle, XCircle, Clock, Loader2 } from 'lucide-react';

function getCategoryStyle(cat: string) {
  switch (cat) {
    case 'A': return 'bg-sw-healthy-bg border-sw-healthy-border text-sw-healthy';
    case 'B': return 'bg-sw-good-bg border-sw-good-border text-sw-good';
    case 'C': return 'bg-sw-warning-bg border-sw-warning-border text-sw-warning';
    case 'D': return 'bg-sw-danger-bg border-sw-danger-border text-sw-danger';
    case 'E': return 'bg-sw-critical-bg border-sw-critical-border text-sw-critical';
    default: return 'bg-sw-offline-bg border-sw-offline-border text-sw-offline';
  }
}

export default function OperatorDashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();

  const { data, isLoading, error } = useQuery({
    queryKey: ['operator-dashboard'],
    queryFn: () => operatorApi.getDashboard(),
    refetchInterval: 30000,
  });

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  if (error) return (
    <div className="p-6 text-center text-destructive">
      Failed to load dashboard. Please refresh the page.
    </div>
  );

  const stats = [
    { label: 'Total Inverters', value: data?.totalInverters ?? 0, icon: Cpu, color: 'text-primary' },
    { label: 'Healthy (A+B)', value: data?.healthy ?? 0, icon: CheckCircle, color: 'text-sw-healthy' },
    { label: 'Warning (C)', value: data?.warning ?? 0, icon: AlertTriangle, color: 'text-sw-warning' },
    { label: 'Critical (D+E)', value: data?.critical ?? 0, icon: XCircle, color: 'text-sw-critical' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <p className="text-sm text-muted-foreground">Welcome back, {user?.name}</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {stats.map(s => (
          <Card key={s.label} className="rounded-xl shadow-sm">
            <CardContent className="p-5">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">{s.label}</p>
                  <p className="text-3xl font-bold mt-1">{s.value}</p>
                </div>
                <s.icon className={`h-8 w-8 ${s.color} opacity-80`} />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {data?.needsAttention?.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-3">Needs Attention</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {data.needsAttention.map((inv: any) => (
              <Card key={inv.id} className={`rounded-xl border-2 ${inv.current_category === 'E' ? 'animate-pulse-border' : ''}`}>
                <CardContent className="p-4 flex items-center justify-between">
                  <div>
                    <p className="font-mono font-bold text-sm">{inv.name}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">
                      {inv.plant_name} › {inv.block_name}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge className={getCategoryStyle(inv.current_category)}>{inv.current_category}</Badge>
                    <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => navigate(`/operator/plant/${inv.plant_id}/block/${inv.block_id}`)}>
                      View
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      )}

      <div>
        <h2 className="text-lg font-semibold mb-3">Recent Alerts</h2>
        <Card className="rounded-xl shadow-sm">
          <CardContent className="p-0">
            <div className="divide-y">
              {!data?.recentAlerts?.length ? (
                <p className="p-6 text-center text-muted-foreground">No recent alerts</p>
              ) : data.recentAlerts.map((alert: any) => (
                <div key={alert.id} className="flex items-center gap-3 px-4 py-3">
                  <div className={`h-2.5 w-2.5 rounded-full flex-shrink-0 ${alert.type === 'critical' ? 'bg-sw-critical' : alert.type === 'warning' ? 'bg-sw-warning' : 'bg-primary'
                    }`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm truncate">{alert.message}</p>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      <Clock className="h-3 w-3" />
                      {new Date(alert.created_at).toLocaleString()}
                    </div>
                  </div>
                  <Badge variant={alert.acknowledged ? 'secondary' : 'destructive'} className="text-xs">
                    {alert.acknowledged ? 'acknowledged' : 'pending'}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
