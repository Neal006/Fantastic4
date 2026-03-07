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
import { Loader2, Bot, FileDown, ChevronRight, Zap, AlertTriangle } from 'lucide-react';
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
  const [trendParam, setTrendParam] = useState<ParamKey>('dc_voltage');
  const [trendRange, setTrendRange] = useState<'24h' | '48h'>('24h');
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const inverterId = inverter?.id;
  // The GenAI system uses the inverter NAME (e.g. INV-P1-L2-0), not the DB UUID
  const inverterName = inverter?.name || inverter?.id;

  // ── Live readings (Trend tab) ──────────────────────────────────────────────
  const { data: readings = [], isFetching: readingsFetching } = useQuery({
    queryKey: ['inverter-readings', inverterId, trendRange],
    queryFn: () => operatorApi.getReadings(inverterId, trendRange),
    enabled: !!inverterId,
    staleTime: 3_000,
    refetchInterval: 5_000,
  });

  // ── Fault history (History tab) ────────────────────────────────────────────
  const { data: faults = [], isFetching: faultsFetching } = useQuery({
    queryKey: ['inverter-faults', inverterId],
    queryFn: () => operatorApi.getFaults(inverterId),
    enabled: !!inverterId,
    staleTime: 60_000,
  });

  // ── AI Explanation — auto-fetches when sheet opens ───────────────────────
  const { data: explanation, isFetching: explanationFetching, refetch: refetchExplanation, isError: explanationError } = useQuery({
    queryKey: ['ai-explanation', inverterName],
    queryFn: () => chatbotApi.getExplanation(inverterName!),
    enabled: !!inverterName,
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
    name: s.feature || s.label || String(s),
    value: typeof s.shap_value === 'number' ? s.shap_value : typeof s.value === 'number' ? s.value : typeof s === 'number' ? s : 0,
  }));
  const topShap = shapData[0] || null;
  const category = inverter.current_category || inverter.category || 'offline';
  const showShap = ['C', 'D', 'E'].includes(category) && shapData.length > 0;

  // ── Trend chart data (downsample to max 300 points for performance) ────────
  const rawTrend = (readings as any[]).map((rd: any) => ({
    time: new Date(rd.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    value: rd[trendParam] != null ? Number(Number(rd[trendParam]).toFixed(2)) : null,
  })).filter(d => d.value != null);
  const MAX_POINTS = 300;
  const trendData = rawTrend.length <= MAX_POINTS
    ? rawTrend
    : rawTrend.filter((_: any, i: number) => i % Math.ceil(rawTrend.length / MAX_POINTS) === 0);
  const tickInterval = trendData.length > 0 ? Math.max(Math.floor(trendData.length / 6) - 1, 0) : 0;

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

        <Tabs defaultValue="overview" className="mt-2">
          <TabsList className="w-full grid grid-cols-4">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="shap" disabled={!showShap}>SHAP</TabsTrigger>
            <TabsTrigger value="trend">Trend</TabsTrigger>
            <TabsTrigger value="history">History</TabsTrigger>
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

            {/* ── Inline AI Summary ── */}
            <div className="rounded-lg border bg-muted/40 overflow-hidden">
              <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/60">
                <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                  <Bot className="h-3.5 w-3.5" />
                  AI Analysis
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 text-xs px-2 gap-1"
                  onClick={() => refetchExplanation()}
                  disabled={explanationFetching}
                >
                  {explanationFetching ? <Loader2 className="h-3 w-3 animate-spin" /> : <Zap className="h-3 w-3" />}
                  {explanation ? 'Refresh' : 'Retry'}
                </Button>
              </div>

              {/* Loading */}
              {explanationFetching && (
                <div className="p-3 space-y-2">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-5/6" />
                  <Skeleton className="h-3 w-4/6" />
                </div>
              )}

              {/* Error */}
              {explanationError && !explanationFetching && (
                <div className="flex items-center gap-2 p-3 text-xs text-destructive">
                  <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
                  Could not load AI analysis. Ensure the GenAI server is running.
                </div>
              )}

              {/* Content */}
              {explanation && !explanationFetching && (
                <div className="p-3 space-y-3 text-sm">
                  {/* Risk + urgency */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xl font-extrabold">{Math.round(explanation.risk_score * 100)}%</span>
                    <span className="text-xs text-muted-foreground">risk score</span>
                    <Badge className={URGENCY_COLORS[explanation.urgency] ?? 'bg-muted'}>
                      {URGENCY_LABEL[explanation.urgency] ?? explanation.urgency.toUpperCase()}
                    </Badge>
                  </div>

                  {/* Summary */}
                  <p className="text-xs text-muted-foreground leading-relaxed">{explanation.summary}</p>

                  {/* Key factors */}
                  {explanation.key_factors?.length > 0 && (
                    <div className="rounded bg-background/70 p-2 space-y-1">
                      <p className="text-xs font-semibold">Key Risk Factors</p>
                      <ul className="space-y-0.5">
                        {explanation.key_factors.map((f: any, i: number) => (
                          <li key={i} className="flex items-start gap-1 text-xs text-muted-foreground">
                            <ChevronRight className="h-3 w-3 flex-shrink-0 mt-0.5" />
                            <span><span className="font-medium text-foreground">{f.feature}</span>{f.raw_value ? ` (${f.raw_value})` : ''}{f.impact ? ` — ${f.impact}` : ''}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Recommended actions */}
                  {explanation.recommended_actions?.length > 0 && (
                    <div>
                      <p className="text-xs font-semibold mb-1">Recommended Actions</p>
                      <ol className="space-y-0.5 list-decimal list-inside text-xs text-muted-foreground">
                        {explanation.recommended_actions.map((a: string, i: number) => (
                          <li key={i}>{a}</li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* PDF download */}
                  {downloadError && (
                    <div className="flex items-center gap-1.5 text-xs text-destructive bg-destructive/10 border border-destructive/20 rounded p-2">
                      <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
                      {downloadError}
                    </div>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    className="w-full gap-2 h-8 text-xs"
                    disabled={downloading}
                    onClick={async () => {
                      setDownloading(true);
                      setDownloadError(null);
                      try {
                        // Step 1: Generate ticket on GenAI server
                        await chatbotApi.generateTicket(inverterName!);
                        // Step 2: Download the PDF
                        const token = sessionStorage.getItem('sw_token');
                        const res = await fetch(chatbotApi.getPdfUrl(inverterName!), {
                          headers: { Authorization: `Bearer ${token}` },
                          credentials: 'include',
                        });
                        if (!res.ok) throw new Error(`PDF not available (${res.status}). Ensure the GenAI server is running.`);
                        const blob = await res.blob();
                        if (blob.type !== 'application/pdf' && blob.size < 100) throw new Error('Invalid PDF received.');
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `${inverterName}-maintenance-ticket.pdf`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);
                      } catch (err: any) {
                        setDownloadError(err?.message || 'Download failed. Please try again.');
                      } finally {
                        setDownloading(false);
                      }
                    }}
                  >
                    {downloading ? (
                      <>
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        Generating ticket…
                      </>
                    ) : (
                      <>
                        <FileDown className="h-3.5 w-3.5" />
                        Download Maintenance Ticket
                      </>
                    )}
                  </Button>
                </div>
              )}

              {/* Empty — waiting for first load */}
              {!explanation && !explanationFetching && !explanationError && (
                <p className="text-xs text-muted-foreground text-center py-4">Generating AI analysis…</p>
              )}
            </div>
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
                  <LineChart data={trendData} margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                    <XAxis dataKey="time" tick={{ fontSize: 9 }} interval={tickInterval} angle={-30} textAnchor="end" height={40} />
                    <YAxis tick={{ fontSize: 11 }} width={55} domain={['auto', 'auto']} />
                    <Tooltip formatter={(v: any) => [`${Number(v).toFixed(2)} ${trendLabel?.unit}`, trendLabel?.label]} />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="hsl(234, 89%, 63%)"
                      strokeWidth={1.5}
                      dot={false}
                      activeDot={{ r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
            <p className="text-xs text-muted-foreground text-center">
              {trendLabel?.label} ({trendLabel?.unit}) — {trendRange === '24h' ? 'Last 24 hours' : 'Last 48 hours'} · {rawTrend.length} readings{rawTrend.length > MAX_POINTS ? ` (showing ${trendData.length})` : ''}
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


        </Tabs>
      </SheetContent>
    </Sheet>
  );
}
