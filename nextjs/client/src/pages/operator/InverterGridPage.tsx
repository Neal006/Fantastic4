import { useState, useEffect } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { operatorApi } from '@/lib/api';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Search, Loader2, RefreshCw } from 'lucide-react';
import InverterDetailSheet from '@/components/InverterDetailSheet';
import { toast } from 'sonner';

type Filter = 'all' | 'healthy' | 'warning' | 'critical' | 'offline';

const categoryCardClass: Record<string, string> = {
  A: 'bg-sw-healthy-bg border-sw-healthy-border',
  B: 'bg-sw-good-bg border-sw-good-border',
  C: 'bg-sw-warning-bg border-sw-warning-border',
  D: 'bg-sw-danger-bg border-sw-danger-border',
  E: 'bg-sw-critical-bg border-sw-critical-border',
  offline: 'bg-sw-offline-bg border-dashed border-sw-offline-border',
};

const categoryTextClass: Record<string, string> = {
  A: 'text-sw-healthy',
  B: 'text-sw-good',
  C: 'text-sw-warning',
  D: 'text-sw-danger',
  E: 'text-sw-critical',
  offline: 'text-sw-offline',
};

function matchesFilter(inv: any, filter: Filter): boolean {
  if (filter === 'all') return true;
  if (filter === 'healthy') return inv.current_category === 'A' || inv.current_category === 'B';
  if (filter === 'warning') return inv.current_category === 'C';
  if (filter === 'critical') return inv.current_category === 'D' || inv.current_category === 'E';
  if (filter === 'offline') return inv.current_category === 'offline' || !inv.is_online;
  return true;
}

function timeAgo(ts: string): string {
  if (!ts) return 'Unknown';
  const mins = Math.floor((Date.now() - new Date(ts).getTime()) / 60000);
  if (mins < 1) return 'Just now';
  if (mins < 60) return `${mins} min ago`;
  return `${Math.floor(mins / 60)}h ${mins % 60}m ago`;
}

// No adapter needed — new InverterDetailSheet uses raw API field names directly

export default function InverterGridPage() {
  const { plantId, blockId } = useParams<{ plantId: string; blockId: string }>();
  const location = useLocation();
  const [filter, setFilter] = useState<Filter>('all');
  const [search, setSearch] = useState('');
  const [selectedInv, setSelectedInv] = useState<any | null>(null);
  const qc = useQueryClient();

  const { data: inverters = [], isLoading, refetch } = useQuery({
    queryKey: ['inverter-grid', plantId, blockId],
    queryFn: () => operatorApi.getInverterGrid(plantId!, blockId!),
    refetchInterval: 30000,
    enabled: !!plantId && !!blockId,
  });

  // Auto-open the inverter sheet when navigated from an alert
  useEffect(() => {
    const targetId = (location.state as any)?.selectInverterId;
    if (!targetId || !inverters.length) return;
    const match = (inverters as any[]).find((inv: any) => inv.id === targetId);
    if (match) {
      setSelectedInv(match);
      // Clear the state so refresh doesn't re-open it
      window.history.replaceState({}, '');
    }
  }, [inverters, location.state]);

  // Fetch plant/block names for the sheet header
  const { data: plants = [] } = useQuery({
    queryKey: ['operator-plants-for-grid'],
    queryFn: () => operatorApi.getPlants(),
    staleTime: 120_000,
  });
  const plantObj = (plants as any[]).find((p: any) => p.id === plantId);
  const blockObj = plantObj?.blocks?.find((b: any) => b.id === blockId);
  const plantName = plantObj?.name ?? `Plant ${plantId}`;
  const blockName = blockObj?.name ?? `Block ${blockId}`;

  const filtered = inverters
    .filter((inv: any) => matchesFilter(inv, filter))
    .filter((inv: any) => inv.name.toLowerCase().includes(search.toLowerCase()) || inv.id.toLowerCase().includes(search.toLowerCase()));

  const filters: { key: Filter; label: string }[] = [
    { key: 'all', label: 'All' },
    { key: 'healthy', label: 'Healthy' },
    { key: 'warning', label: 'Warning' },
    { key: 'critical', label: 'Critical' },
    { key: 'offline', label: 'Offline' },
  ];

  if (!plantId || !blockId) return <p>Invalid route</p>;
  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-1.5 text-sm text-muted-foreground mb-1">
            <span>Dashboard</span><span>›</span><span>Plant</span><span>›</span>
            <span className="text-foreground font-medium">Block Inverters</span>
          </div>
          <h1 className="text-2xl font-bold">Inverter Grid</h1>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} className="gap-1.5">
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {filters.map(f => (
          <Button
            key={f.key}
            variant={filter === f.key ? 'default' : 'outline'}
            size="sm"
            onClick={() => setFilter(f.key)}
            className="text-xs"
          >
            {f.label}
          </Button>
        ))}
        <div className="relative ml-auto">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
          <Input
            placeholder="Search inverter..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-8 h-8 w-48 text-xs"
          />
        </div>
      </div>

      {filtered.length === 0 ? (
        <div className="text-center py-20 text-muted-foreground">
          <p className="text-lg font-medium">No inverters found</p>
          <p className="text-sm mt-1">Try adjusting your filters</p>
        </div>
      ) : (
        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))' }}>
          {filtered.map((inv: any) => (
            <button
              key={inv.id}
              onClick={() => setSelectedInv(inv)}
              className={`p-4 rounded-xl border-2 text-left transition-all hover:shadow-md ${categoryCardClass[inv.current_category] ?? categoryCardClass.offline} ${(inv.current_category === 'D' || inv.current_category === 'E') ? 'animate-pulse-border' : ''
                }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-sm font-bold">{inv.name}</span>
                <Badge className={`${categoryTextClass[inv.current_category] ?? ''} border-current bg-transparent text-xs font-bold`}>
                  {inv.current_category === 'offline' ? 'OFF' : inv.current_category}
                </Badge>
              </div>
              <p className={`text-3xl font-extrabold ${categoryTextClass[inv.current_category] ?? ''}`}>
                {inv.current_category === 'offline' ? '—' : inv.current_category}
              </p>
              <p className="text-[10px] text-muted-foreground mt-2">
                {timeAgo(inv.last_data_at)}
              </p>
            </button>
          ))}
        </div>
      )}

      <InverterDetailSheet
        inverter={selectedInv}
        onClose={() => setSelectedInv(null)}
        plantName={plantName}
        blockName={blockName}
      />
    </div>
  );
}
