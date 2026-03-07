import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Building2, Layers, Cpu, Users, AlertTriangle, WifiOff, Clock, Loader2, ShieldCheck, ShieldAlert, ShieldX } from 'lucide-react';

export default function AdminDashboard() {
  const { data, isLoading } = useQuery({
    queryKey: ['admin-dashboard'],
    queryFn: () => adminApi.getDashboard(),
    refetchInterval: 5000, // Update every 5 seconds for real-time data
  });

  const { data: auditLogs = [] } = useQuery({
    queryKey: ['audit-logs-preview'],
    queryFn: () => adminApi.getAuditLogs({ limit: '5' }),
  });

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  const stats = [
    { label: 'Total Plants', value: data?.totalPlants ?? 0, icon: Building2, color: 'text-primary' },
    { label: 'Total Blocks', value: data?.totalBlocks ?? 0, icon: Layers, color: 'text-primary' },
    { label: 'Total Inverters', value: data?.totalInverters ?? 0, icon: Cpu, color: 'text-primary' },
    { label: 'Total Operators', value: data?.totalOperators ?? 0, icon: Users, color: 'text-primary' },
    { label: 'Faults Active', value: data?.faultCount ?? 0, icon: AlertTriangle, color: 'text-sw-critical' },
    { label: 'Offline', value: data?.offlineCount ?? 0, icon: WifiOff, color: 'text-sw-offline' },
  ];

  const categoryBreakdown = data?.categoryBreakdown || { A: 0, B: 0, C: 0, D: 0, E: 0, offline: 0 };
  
  const categoryCards = [
    { 
      category: 'A', 
      label: 'No Risk', 
      value: categoryBreakdown.A, 
      icon: ShieldCheck, 
      color: 'text-emerald-500',
      bgColor: 'bg-emerald-500/10',
      borderColor: 'border-emerald-500/20'
    },
    { 
      category: 'B', 
      label: 'Degradation Risk (Low)', 
      value: categoryBreakdown.B, 
      icon: ShieldAlert, 
      color: 'text-orange-400',
      bgColor: 'bg-orange-500/10',
      borderColor: 'border-orange-500/20'
    },
    { 
      category: 'C', 
      label: 'Degradation Risk', 
      value: categoryBreakdown.C, 
      icon: ShieldAlert, 
      color: 'text-orange-500',
      bgColor: 'bg-orange-500/10',
      borderColor: 'border-orange-500/20'
    },
    { 
      category: 'D', 
      label: 'Shutdown Risk (High)', 
      value: categoryBreakdown.D, 
      icon: ShieldX, 
      color: 'text-red-500',
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-500/20'
    },
    { 
      category: 'E', 
      label: 'Shutdown Risk (Critical)', 
      value: categoryBreakdown.E, 
      icon: ShieldX, 
      color: 'text-red-600',
      bgColor: 'bg-red-500/10',
      borderColor: 'border-red-500/20'
    },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Admin Dashboard</h1>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4">
        {stats.map(s => (
          <Card key={s.label} className="rounded-xl shadow-sm">
            <CardContent className="p-4 text-center">
              <s.icon className={`h-6 w-6 mx-auto mb-2 ${s.color} opacity-80`} />
              <p className="text-2xl font-bold">{s.value}</p>
              <p className="text-xs text-muted-foreground mt-1">{s.label}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-3">Inverter Status Breakdown (Real-time)</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {categoryCards.map(cat => (
            <Card key={cat.category} className={`rounded-xl shadow-sm border-2 ${cat.borderColor} ${cat.bgColor}`}>
              <CardContent className="p-4">
                <div className="flex items-center gap-3 mb-2">
                  <cat.icon className={`h-5 w-5 ${cat.color}`} />
                  <Badge variant="outline" className={`${cat.color} border-current`}>
                    Category {cat.category}
                  </Badge>
                </div>
                <p className="text-3xl font-bold mb-1">{cat.value}</p>
                <p className="text-xs text-muted-foreground">{cat.label}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-3">System Status</h2>
        <Card className="rounded-xl shadow-sm">
          <CardContent className="p-4 flex flex-wrap gap-4">
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-sw-healthy" />
              <span className="text-sm">Data Pipeline: Online</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full bg-sw-healthy" />
              <span className="text-sm">AI Model: Responsive</span>
            </div>
            {data?.lastDataAt && (
              <div className="text-xs text-muted-foreground ml-auto">
                Last data: {new Date(data.lastDataAt).toLocaleString()}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-3">Recent Admin Activity</h2>
        <Card className="rounded-xl shadow-sm">
          <CardContent className="p-0 divide-y">
            {(auditLogs as any[]).length === 0 ? (
              <p className="p-6 text-center text-muted-foreground text-sm">No recent activity</p>
            ) : (auditLogs as any[]).map((log: any) => (
              <div key={log.id} className="px-4 py-3 flex items-center gap-3">
                <Clock className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                <div className="flex-1">
                  <p className="text-sm">
                    {log.action.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase())}
                    {log.details && typeof log.details === 'object'
                      ? ` — ${Object.values(log.details).join(', ')}`
                      : ''}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(log.created_at).toLocaleString()} • {log.user_name || log.user_id}
                  </p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
