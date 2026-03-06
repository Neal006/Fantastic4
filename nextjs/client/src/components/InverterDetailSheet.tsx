import { Sheet, SheetContent, SheetHeader, SheetTitle } from '@/components/ui/sheet';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, LineChart, Line,
} from 'recharts';
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { MessageCircle, Loader2, Bot, FileDown, ChevronRight, Zap } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { operatorApi, chatbotApi } from '@/lib/api';

interface Props {
  inverter: any | null;  // real API inverter shape
  onClose: () => void;
  plantName: string;
  blockName: string;
}

const catBadgeClass: Record<string, string> = {
  A: 'bg-sw-healthy text-primary-foreground',
  B: 'bg-sw-good text-primary-foreground',
  C: 'bg-sw-warning text-primary-foreground',
  D: 'bg-sw-danger text-primary-foreground',
  E: 'bg-sw-critical text-primary-foreground',
  offline: 'bg-sw-offline text-primary-foreground',
};

// Field key → {label, unit, dbColumn}
const PARAMS = [
  { key: 'dc_voltage', label: 'DC Voltage', unit: 'V' },
  { key: 'dc_current', label: 'DC Current', unit: 'A' },
  { key: 'ac_power', label: 'AC Power', unit: 'kW' },
  { key: 'module_temp', label: 'Module Temp', unit: '°C' },
  { key: 'ambient_temp', label: 'Ambient Temp', unit: '°C' },
  { key: 'irradiation', label: 'Irradiation', unit: 'W/m²' },
] as const;
type ParamKey = typeof PARAMS[number]['key'];

const URGENCY_COLORS: Record<string, string> = {
  immediate: 'bg-red-500 text-white',
  within_24h: 'bg-orange-500 text-white',
  scheduled: 'bg-yellow-500 text-black',
  routine: 'bg-green-500 text-white',
};

const URGENCY_LABEL: Record<string, string> = {
  immediate: 'IMMEDIATE',
  within_24h: 'WITHIN 24H',
  scheduled: 'SCHEDULED',
  routine: 'ROUTINE',
};

export default function InverterDetailSheet({ inverter, onClose, plantName, blockName }: Props) {
  const navigate = useNavigate();
  const [trendParam, setTrendParam] = useState<ParamKey>('dc_voltage');
  const [trendRange, setTrendRange] = useState<'24h' | '48h'>('24h');
  const [aiTabActive, setAiTabActive] = useState(false);

  const inverterId = inverter?.id;
  // The GenAI system uses the inverter NAME (e.g. INV-P1-L2-0), not the DB UUID
  const inverterName = inverter?.name || inverter?.id;

  // ── Live readings (Trend tab) ──────────────────────────────────────────────
  const { data: readings = [], isFetching: readingsFetching } = useQuery({
    queryKey: ['inverter-readings', inverterId, trendRange],
    queryFn: () => operatorApi.getReadings(inverterId, trendRange),
    enabled: !!inverterId,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });

  // ── Fault history (History tab) ────────────────────────────────────────────
  const { data: faults = [], isFetching: faultsFetching } = useQuery({
    queryKey: ['inverter-faults', inverterId],
    queryFn: () => operatorApi.getFaults(inverterId),
    enabled: !!inverterId,
    staleTime: 60_000,
  });

  // ── AI Explanation (AI tab) ────────────────────────────────────────────────
  const { data: explanation, isFetching: explanationFetching, refetch: refetchExplanation, isError: explanationError } = useQuery({
    queryKey: ['ai-explanation', inverterName],
    queryFn: () => chatbotApi.getExplanation(inverterName!),
    enabled: !!inverterName && aiTabActive,
    staleTime: 5 * 60_000,
    retry: 1,
  });

  if (!inverter) return null;

  // ── Overview: current readings from the latest reading on the inverter ─────
  const r = inverter; // API returns dc_voltage etc directly on the inverter object
  const overviewReadings = PARAMS.map(p => ({
    label: p.label,
    unit: p.unit,
    value: r[p.key] != null ? Number(r[p.key]).toFixed(2) : '—',
  }));

  // ── SHAP tab ──────────────────────────────────────────────────────────────
  const shapRaw: any[] = Array.isArray(r.shap_values) ? r.shap_values : [];
  const shapData = shapRaw.map((s: any) => ({
    name: s.label || s.feature || String(s),
    value: typeof s.value === 'number' ? s.value : typeof s === 'number' ? s : 0,
  }));
  const topShap = shapData[0] || null;
  const category = inverter.current_category || inverter.category || 'offline';
  const showShap = ['C', 'D', 'E'].includes(category) && shapData.length > 0;

  // ── Trend chart data ───────────────────────────────────────────────────────
  const trendData = (readings as any[]).map((rd: any) => ({
    time: new Date(rd.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    value: rd[trendParam] != null ? Number(Number(rd[trendParam]).toFixed(2)) : null,
  })).filter(d => d.value != null);

  const trendLabel = PARAMS.find(p => p.key === trendParam);

  return (
    <Sheet open={!!inverter} onOpenChange={() => onClose()}>
      <SheetContent className="w-full sm:max-w-xl overflow-y-auto">
        <SheetHeader className="pb-4">
          <div className="flex items-center gap-2">
            <SheetTitle className="font-mono">{inverter.name || inverter.id}</SheetTitle>
            <Badge className={catBadgeClass[category] ?? 'bg-muted'}>{category}</Badge>
          </div>
          <p className="text-sm text-muted-foreground">{plantName} › {blockName}</p>
        </SheetHeader>

        <Tabs defaultValue="overview" className="mt-2" onValueChange={v => { if (v === 'ai') setAiTabActive(true); }}>
          <TabsList className="w-full grid grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="shap" disabled={!showShap}>SHAP</TabsTrigger>
            <TabsTrigger value="trend">Trend</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
            <TabsTrigger value="ai" className="flex items-center gap-1">
              <Bot className="h-3 w-3" />AI
            </TabsTrigger>
          </TabsList>

          {/* ── Overview ── */}
          <TabsContent value="overview" className="space-y-4 mt-4">
            <div className="grid grid-cols-2 gap-3">
              {overviewReadings.map(rd => (
                <div key={rd.label} className="p-3 rounded-lg bg-muted">
                  <p className="text-xs text-muted-foreground">{rd.label}</p>
                  <p className="text-lg font-bold">{rd.value} <span className="text-xs font-normal text-muted-foreground">{rd.unit}</span></p>
                </div>
              ))}
            </div>

            {inverter.confidence != null && (
              <div className="p-3 rounded-lg bg-muted">
                <div className="flex items-center justify-between mb-2">
                  <p className="text-xs text-muted-foreground">AI Confidence</p>
                  <p className="text-sm font-bold">{(Number(inverter.confidence) * 100).toFixed(0)}%</p>
                </div>
                <Progress value={Number(inverter.confidence) * 100} className="h-2" />
              </div>
            )}

            {inverter.fault_type && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                <p className="text-xs text-muted-foreground">Fault Type</p>
                <p className="text-sm font-semibold text-destructive">{inverter.fault_type}</p>
              </div>
            )}

            <Button
              variant="outline"
              className="w-full"
              onClick={() => { onClose(); navigate('/operator/chatbot', { state: { context: { inverter_id: inverter.id, inverter_name: inverter.name, category, fault_type: inverter.fault_type } } }); }}
            >
              <MessageCircle className="h-4 w-4 mr-2" />
              Ask AI about this inverter
            </Button>
          </TabsContent>

          {/* ── SHAP ── */}
          <TabsContent value="shap" className="mt-4 space-y-4">
            {shapData.length === 0 ? (
              <p className="text-center text-muted-foreground py-8 text-sm">No SHAP data available for this inverter</p>
            ) : (
              <>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={shapData} layout="vertical" margin={{ left: 100, right: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={100} />
                      <Tooltip formatter={(v: any) => Number(v).toFixed(4)} />
                      <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {shapData.map((entry, i) => (
                          <Cell key={i} fill={entry.value >= 0 ? 'hsl(0, 84%, 60%)' : 'hsl(142, 71%, 45%)'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                {topShap && (
                  <div className="p-3 rounded-lg bg-muted text-sm">
                    <p>
                      <span className="font-semibold">Primary fault contributor:</span> {topShap.name} ({(Math.abs(topShap.value) * 100).toFixed(0)}% impact)
                      {topShap.value > 0 ? ' — value is significantly above normal range' : ' — value is below expected range'}
                    </p>
                  </div>
                )}
              </>
            )}
          </TabsContent>

          {/* ── Trend ── */}
          <TabsContent value="trend" className="mt-4 space-y-3">
            <div className="flex gap-2 flex-wrap">
              <Select value={trendParam} onValueChange={v => setTrendParam(v as ParamKey)}>
                <SelectTrigger className="w-44 h-8 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PARAMS.map(p => (
                    <SelectItem key={p.key} value={p.key}>{p.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={trendRange} onValueChange={v => setTrendRange(v as any)}>
                <SelectTrigger className="w-28 h-8 text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="24h">Last 24h</SelectItem>
                  <SelectItem value="48h">Last 48h</SelectItem>
                </SelectContent>
              </Select>
              {readingsFetching && <Loader2 className="h-4 w-4 animate-spin self-center text-muted-foreground" />}
            </div>

            <div className="h-64">
              {trendData.length === 0 ? (
                <div className="h-full flex items-center justify-center rounded-lg border-2 border-dashed border-muted">
                  <p className="text-sm text-muted-foreground">
                    {readingsFetching ? 'Loading...' : 'No readings recorded yet — the simulator generates data every 15s'}
                  </p>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trendData} margin={{ top: 4, right: 16, bottom: 4, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 11 }} width={50} />
                    <Tooltip formatter={(v: any) => [`${Number(v).toFixed(2)} ${trendLabel?.unit}`, trendLabel?.label]} />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="hsl(234, 89%, 63%)"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
            <p className="text-xs text-muted-foreground text-center">
              {trendLabel?.label} ({trendLabel?.unit}) — {trendRange === '24h' ? 'Last 24 hours' : 'Last 48 hours'} · {trendData.length} readings
            </p>
          </TabsContent>

          {/* ── History ── */}
          <TabsContent value="history" className="mt-4">
            {faultsFetching ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
              </div>
            ) : (faults as any[]).length === 0 ? (
              <p className="text-center text-muted-foreground py-8">No fault history for this inverter</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Timestamp</TableHead>
                    <TableHead>Category</TableHead>
                    <TableHead>Fault Type</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(faults as any[]).map((ev: any, i) => (
                    <TableRow key={i}>
                      <TableCell className="text-xs">{new Date(ev.timestamp).toLocaleString()}</TableCell>
                      <TableCell><Badge className={catBadgeClass[ev.category] || 'bg-muted'}>{ev.category}</Badge></TableCell>
                      <TableCell className="text-sm">{ev.fault_type || '—'}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </TabsContent>

          {/* ── AI Explanation ── */}
          <TabsContent value="ai" className="mt-4 space-y-4">
            {/* Action bar */}
            <div className="flex items-center justify-between">
              <p className="text-xs text-muted-foreground">Powered by SolarGuard AI (GenAI)</p>
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs gap-1"
                onClick={() => refetchExplanation()}
                disabled={explanationFetching}
              >
                {explanationFetching ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
                {explanation ? 'Refresh' : 'Explain'}
              </Button>
            </div>

            {/* Loading skeletons */}
            {explanationFetching && (
              <div className="space-y-3">
                <Skeleton className="h-5 w-32" />
                <Skeleton className="h-20 w-full" />
                <Skeleton className="h-4 w-24" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-4/6" />
              </div>
            )}

            {/* Error state */}
            {explanationError && !explanationFetching && (
              <div className="rounded-lg bg-destructive/10 border border-destructive/20 p-4 text-sm text-destructive">
                Could not load AI explanation. Make sure the GenAI server is running on port 8000.
              </div>
            )}

            {/* Explanation content */}
            {explanation && !explanationFetching && (
              <div className="space-y-4 text-sm">
                {/* Risk score + urgency */}
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-2xl font-extrabold">
                    {Math.round(explanation.risk_score * 100)}%
                  </span>
                  <span className="text-muted-foreground text-xs">risk score</span>
                  <Badge className={URGENCY_COLORS[explanation.urgency] ?? 'bg-muted'}>
                    {URGENCY_LABEL[explanation.urgency] ?? explanation.urgency.toUpperCase()}
                  </Badge>
                </div>

                {/* Summary */}
                <p className="text-muted-foreground leading-relaxed">{explanation.summary}</p>

                {/* Key factors */}
                {explanation.key_factors?.length > 0 && (
                  <div className="rounded-lg bg-muted p-3 space-y-2">
                    <p className="font-semibold">Key Risk Factors</p>
                    <ul className="space-y-1.5">
                      {explanation.key_factors.map((f: any, i: number) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 mt-0.5 text-muted-foreground" />
                          <span>
                            <span className="font-medium">{f.feature}</span>
                            {f.raw_value ? ` (${f.raw_value})` : ''}
                            {f.impact ? ` — ${f.impact}` : ''}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommended actions */}
                {explanation.recommended_actions?.length > 0 && (
                  <div>
                    <p className="font-semibold mb-2">Recommended Actions</p>
                    <ol className="space-y-1.5 list-decimal list-inside text-muted-foreground">
                      {explanation.recommended_actions.map((a: string, i: number) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Disclaimer */}
                {explanation.disclaimer && (
                  <p className="text-[10px] text-muted-foreground italic border-t pt-3">{explanation.disclaimer}</p>
                )}
              </div>
            )}

            {/* Empty state */}
            {!explanation && !explanationFetching && !explanationError && (
              <p className="text-sm text-muted-foreground text-center py-6">
                Click <strong>Explain</strong> to fetch an AI-generated risk analysis.
              </p>
            )}

            {/* PDF Download */}
            <div className="border-t pt-4">
              <Button
                variant="outline"
                size="sm"
                className="w-full gap-2"
                onClick={async () => {
                  try {
                    await chatbotApi.generateTicket(inverterName!);
                    const token = sessionStorage.getItem('sw_token');
                    const res = await fetch(chatbotApi.getPdfUrl(inverterName!), {
                      headers: { Authorization: `Bearer ${token}` },
                      credentials: 'include',
                    });
                    if (!res.ok) throw new Error('PDF unavailable');
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = `${inverterName}-ticket.pdf`; a.click();
                    URL.revokeObjectURL(url);
                  } catch { /* silently fail */ }
                }}
              >
                <FileDown className="h-4 w-4" /> Download Maintenance Ticket PDF
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}
