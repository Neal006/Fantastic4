import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Button } from '@/components/ui/button';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';

const DEFAULT_THRESHOLDS = { A: [90, 100], B: [75, 89], C: [50, 74], D: [25, 49], E: [0, 24] };

export default function SettingsPage() {
  const qc = useQueryClient();

  const { data: settings, isLoading } = useQuery({
    queryKey: ['admin-settings'],
    queryFn: () => adminApi.getSettings(),
  });

  const [thresholds, setThresholds] = useState<Record<string, [number, number]>>(DEFAULT_THRESHOLDS);
  const [staleTimeout, setStaleTimeout] = useState(10);

  // Sync state when settings load
  useState(() => {
    if (settings) {
      if (settings.thresholds) setThresholds(settings.thresholds);
      if (settings.staleTimeoutMinutes) setStaleTimeout(settings.staleTimeoutMinutes);
    }
  });

  const saveMutation = useMutation({
    mutationFn: () => adminApi.updateSettings({ staleTimeoutMinutes: staleTimeout, thresholds }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-settings'] });
      toast.success('Settings saved successfully');
    },
    onError: (e: any) => toast.error(e.message),
  });

  const updateThreshold = (cat: 'A' | 'B' | 'C' | 'D' | 'E', idx: 0 | 1, val: string) => {
    setThresholds(prev => ({ ...prev, [cat]: prev[cat].map((v: number, i: number) => i === idx ? Number(val) : v) as [number, number] }));
  };

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  const activeThresholds = settings?.thresholds || thresholds;
  const activeTimeout = settings?.staleTimeoutMinutes ?? staleTimeout;

  return (
    <div className="space-y-6 max-w-2xl">
      <h1 className="text-2xl font-bold">Settings</h1>

      <Card className="rounded-xl shadow-sm">
        <CardHeader><CardTitle className="text-lg">Category Thresholds (%)</CardTitle></CardHeader>
        <CardContent className="space-y-3">
          <p className="text-xs text-muted-foreground">Define the confidence score range for each AI category grade.</p>
          {(['A', 'B', 'C', 'D', 'E'] as const).map(cat => (
            <div key={cat} className="flex items-center gap-3">
              <span className="font-bold w-8">{cat}</span>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  className="w-20 h-8 text-sm"
                  defaultValue={activeThresholds[cat]?.[0] ?? thresholds[cat][0]}
                  onChange={e => updateThreshold(cat, 0, e.target.value)}
                />
                <span className="text-muted-foreground">–</span>
                <Input
                  type="number"
                  className="w-20 h-8 text-sm"
                  defaultValue={activeThresholds[cat]?.[1] ?? thresholds[cat][1]}
                  onChange={e => updateThreshold(cat, 1, e.target.value)}
                />
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card className="rounded-xl shadow-sm">
        <CardHeader><CardTitle className="text-lg">Stale Data Timeout</CardTitle></CardHeader>
        <CardContent>
          <p className="text-xs text-muted-foreground mb-3">Inverters not reporting within this window are considered stale.</p>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              className="w-24 h-8"
              defaultValue={activeTimeout}
              onChange={e => setStaleTimeout(Number(e.target.value))}
            />
            <span className="text-sm text-muted-foreground">minutes</span>
          </div>
        </CardContent>
      </Card>

      <Button onClick={() => saveMutation.mutate()} disabled={saveMutation.isPending}>
        {saveMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
        Save Settings
      </Button>
    </div>
  );
}
