import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Badge } from '@/components/ui/badge';
import { Loader2 } from 'lucide-react';

const categoryCardClass: Record<string, string> = {
  A: 'bg-sw-healthy-bg border-sw-healthy-border',
  B: 'bg-sw-good-bg border-sw-good-border',
  C: 'bg-sw-warning-bg border-sw-warning-border',
  D: 'bg-sw-danger-bg border-sw-danger-border',
  E: 'bg-sw-critical-bg border-sw-critical-border',
  offline: 'bg-sw-offline-bg border-dashed border-sw-offline-border',
};
const categoryTextClass: Record<string, string> = {
  A: 'text-sw-healthy', B: 'text-sw-good', C: 'text-sw-warning',
  D: 'text-sw-danger', E: 'text-sw-critical', offline: 'text-sw-offline',
};

// Inner component: loads inverters per block lazily
function BlockGrid({ blockId, blockName }: { blockId: string; blockName: string }) {
  const { data: inverters = [], isLoading } = useQuery({
    queryKey: ['admin-inverters-monitor', blockId],
    queryFn: () => adminApi.getInverters(blockId),
    refetchInterval: 15000,
  });

  if (isLoading) return <div className="flex items-center gap-2 py-2"><Loader2 className="h-3 w-3 animate-spin" /><span className="text-xs text-muted-foreground">Loading {blockName}...</span></div>;

  return (
    <div>
      <h3 className="text-sm font-medium text-muted-foreground mb-2">{blockName}</h3>
      <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}>
        {(inverters as any[]).map((inv: any) => {
          const cat = inv.current_category || 'offline';
          return (
            <div key={inv.id} className={`p-3 rounded-lg border-2 ${categoryCardClass[cat] ?? categoryCardClass.offline} ${(cat === 'D' || cat === 'E') ? 'animate-pulse-border' : ''}`}>
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-xs font-bold truncate">{inv.name}</span>
                <Badge className={`${categoryTextClass[cat] ?? ''} border-current bg-transparent text-[10px] font-bold`}>
                  {cat === 'offline' ? 'OFF' : cat}
                </Badge>
              </div>
              <p className={`text-xl font-extrabold ${categoryTextClass[cat] ?? ''}`}>{cat === 'offline' ? '—' : cat}</p>
            </div>
          );
        })}
        {(inverters as any[]).length === 0 && <p className="text-xs text-muted-foreground col-span-full py-2">No inverters</p>}
      </div>
    </div>
  );
}

export default function LiveMonitoring() {
  const { data: plants = [], isLoading: plantsLoading } = useQuery({
    queryKey: ['admin-plants'],
    queryFn: () => adminApi.getPlants(),
  });

  if (plantsLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Live Monitoring</h1>
        <p className="text-sm text-muted-foreground">Read-only view of all plants and inverters — refreshes every 15s</p>
      </div>

      {(plants as any[]).filter((p: any) => p.status !== 'decommissioned').map((plant: any) => (
        <PlantSection key={plant.id} plant={plant} />
      ))}

      {(plants as any[]).length === 0 && (
        <p className="text-center text-muted-foreground py-16">No plants configured</p>
      )}
    </div>
  );
}

function PlantSection({ plant }: { plant: any }) {
  const { data: blocks = [] } = useQuery({
    queryKey: ['admin-blocks', plant.id],
    queryFn: () => adminApi.getBlocks(plant.id),
    staleTime: 60000,
  });

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold">
        {plant.name} <span className="text-sm text-muted-foreground font-normal">— {plant.location}</span>
      </h2>
      <div className="space-y-4">
        {(blocks as any[]).map((block: any) => (
          <BlockGrid key={block.id} blockId={block.id} blockName={block.name} />
        ))}
        {(blocks as any[]).length === 0 && <p className="text-sm text-muted-foreground">No blocks in this plant</p>}
      </div>
    </div>
  );
}
