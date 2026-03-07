import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import {
  Loader2, Play, RotateCcw, Zap, AlertTriangle, CheckCircle2,
  ShieldAlert, ShieldCheck, ShieldX, ChevronRight, FileDown, Beaker,
} from 'lucide-react';
import { mlApi } from '@/lib/api';

const catBadge: Record<string, { color: string; label: string; icon: any }> = {
  A: { color: 'bg-emerald-500 text-white', label: 'A — No Risk', icon: ShieldCheck },
  B: { color: 'bg-orange-500 text-white', label: 'B — Degradation Risk', icon: ShieldAlert },
  C: { color: 'bg-orange-500 text-white', label: 'C — Degradation Risk', icon: ShieldAlert },
  D: { color: 'bg-red-500 text-white', label: 'D — Shutdown Risk', icon: ShieldX },
  E: { color: 'bg-red-500 text-white', label: 'E — Shutdown Risk', icon: ShieldX },
};

interface FormData {
  inverter_id: string;
  dc_voltage: string;
  dc_current: string;
  ac_power: string;
  module_temp: string;
  ambient_temp: string;
  irradiation: string;
  alarm_code: string;
  op_state: string;
  power_factor: string;
  frequency: string;
}

const EMPTY: FormData = {
  inverter_id: 'TEST-INV',
  dc_voltage: '', dc_current: '', ac_power: '',
  module_temp: '', ambient_temp: '', irradiation: '',
  alarm_code: '0', op_state: '5120',
  power_factor: '', frequency: '',
};

const PRESETS: { label: string; desc: string; badge: string; data: FormData }[] = [
  {
    label: 'Healthy Inverter',
    desc: 'Normal PV string readings — expect Category A',
    badge: 'A',
    data: {
      inverter_id: 'INV-DEMO-HEALTHY',
      dc_voltage: '36', dc_current: '9.5', ac_power: '8.5',
      module_temp: '42', ambient_temp: '30', irradiation: '800',
      alarm_code: '0', op_state: '5120', power_factor: '0.98', frequency: '50.01',
    },
  },
  {
    label: 'String Degradation (C)',
    desc: 'Low current & power output — expect Category C',
    badge: 'C',
    data: {
      inverter_id: 'INV-DEMO-DEGRAD-C',
      dc_voltage: '40.2', dc_current: '0.4', ac_power: '4.1',
      module_temp: '44.5', ambient_temp: '30.1', irradiation: '815',
      alarm_code: '559', op_state: '5120', power_factor: '0.96', frequency: '49.9',
    },
  },
  {
    label: 'String Degradation (D)',
    desc: 'Low current & power output — expect Category D',
    badge: 'D',
    data: {
      inverter_id: 'INV-DEMO-DEGRAD',
      dc_voltage: '28', dc_current: '2.5', ac_power: '1.8',
      module_temp: '48', ambient_temp: '36', irradiation: '800',
      alarm_code: '0', op_state: '5120', power_factor: '0.91', frequency: '50.00',
    },
  },
  {
    label: 'Overheating Risk',
    desc: 'High module temp with alarm — expect Category E',
    badge: 'E',
    data: {
      inverter_id: 'INV-DEMO-OVERHEAT',
      dc_voltage: '30', dc_current: '4', ac_power: '2.0',
      module_temp: '72', ambient_temp: '48', irradiation: '800',
      alarm_code: '4003', op_state: '0', power_factor: '0.85', frequency: '50.05',
    },
  },
  {
    label: 'Grid Fault (Shutdown)',
    desc: 'Near-zero output, inverter offline — expect Category E',
    badge: 'E',
    data: {
      inverter_id: 'INV-DEMO-GRIDFAULT',
      dc_voltage: '0.5', dc_current: '0.1', ac_power: '0',
      module_temp: '36', ambient_temp: '32', irradiation: '800',
      alarm_code: '5001', op_state: '0', power_factor: '0.35', frequency: '50.00',
    },
  },
];

const FIELDS = [
  { key: 'dc_voltage', label: 'PV String Voltage', unit: 'V', min: 0, max: 100, hint: 'Healthy: 30–42 V' },
  { key: 'dc_current', label: 'PV String Current', unit: 'A', min: 0, max: 30, hint: 'Healthy: 5–12 A' },
  { key: 'ac_power', label: 'Inverter Power', unit: 'kW', min: 0, max: 30, hint: 'Healthy: 4–12 kW' },
  { key: 'module_temp', label: 'Module Temp', unit: '°C', min: -40, max: 150, hint: 'Healthy: 35–55°C' },
  { key: 'ambient_temp', label: 'Ambient Temp', unit: '°C', min: -40, max: 80, hint: 'Healthy: 20–40°C' },
  { key: 'irradiation', label: 'Irradiation', unit: 'W/m²', min: 0, max: 1500, hint: 'Healthy: 780–820' },
] as const;

const OPT_FIELDS = [
  { key: 'alarm_code', label: 'Alarm Code', hint: '0 = no alarm' },
  { key: 'op_state', label: 'Op State', hint: '5120 = running, 0 = off' },
  { key: 'power_factor', label: 'Power Factor', hint: '0.85–0.99' },
  { key: 'frequency', label: 'Frequency', hint: '49.8–50.3 Hz' },
] as const;

export default function MLPredictPage() {
  const [form, setForm] = useState<FormData>({ ...EMPTY });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any | null>(null);

  const set = (key: keyof FormData, val: string) => setForm(prev => ({ ...prev, [key]: val }));

  const loadPreset = (preset: typeof PRESETS[number]) => {
    setForm({ ...preset.data });
    setResult(null);
    setError(null);
  };

  const reset = () => { setForm({ ...EMPTY }); setResult(null); setError(null); };

  const submit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        inverter_id: form.inverter_id || 'TEST-INV',
        dc_voltage: Number(form.dc_voltage),
        dc_current: Number(form.dc_current),
        ac_power: Number(form.ac_power),
        module_temp: Number(form.module_temp),
        ambient_temp: Number(form.ambient_temp),
        irradiation: Number(form.irradiation),
        alarm_code: form.alarm_code !== '' ? Number(form.alarm_code) : 0,
        op_state: form.op_state !== '' ? Number(form.op_state) : 5120,
        power_factor: form.power_factor ? Number(form.power_factor) : null,
        frequency: form.frequency ? Number(form.frequency) : null,
      };

      for (const f of FIELDS) {
        if (!form[f.key] || isNaN(Number(form[f.key]))) {
          setError(`${f.label} is required and must be a number`);
          setLoading(false);
          return;
        }
      }

      const prediction = await mlApi.predict({
        ...payload,
        include_shap: true,
        include_plot: false,
      });
      setResult(prediction);
    } catch (err: any) {
      setError(err?.message || 'Prediction failed. Ensure ML inference server is running on port 8001.');
    } finally {
      setLoading(false);
    }
  };

  const pred = result;
  const cat = pred?.category;
  const catInfo = cat ? catBadge[cat] : null;
  const CatIcon = catInfo?.icon || ShieldCheck;

  const shapRaw: any[] = pred?.shap?.top_features || [];
  const shapData = shapRaw.map((s: any) => ({
    name: s.feature,
    value: s.shap_value ?? 0,
  }));

  const probData = pred?.probabilities
    ? Object.entries(pred.probabilities).map(([k, v]) => ({
        name: k.replace('_', ' '),
        value: Number(((v as number) * 100).toFixed(1)),
      }))
    : [];

  // Input echo — the readings field in ML response should match what we sent
  const inputEcho = pred?.readings || {};
  const inputSent = result ? {
    dc_voltage: Number(form.dc_voltage),
    dc_current: Number(form.dc_current),
    ac_power: Number(form.ac_power),
    module_temp: Number(form.module_temp),
    ambient_temp: Number(form.ambient_temp),
    irradiation: Number(form.irradiation),
  } : {};

  // Clean JSON for display — strip all_values and class_shap (not from our model output)
  const cleanPred = pred ? {
    ...pred,
    shap: pred.shap ? {
      top_features: pred.shap.top_features,
      predicted_class: pred.shap.predicted_class,
      base_value: pred.shap.base_value,
    } : null,
  } : null;

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Beaker className="h-6 w-6 text-primary" />
            ML Prediction Tester
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Enter sensor readings manually and see the XGBoost model's risk prediction with SHAP explanations in real time.
          </p>
        </div>
      </div>

      {/* Example Presets */}
      <div>
        <h2 className="text-sm font-semibold mb-2 text-muted-foreground uppercase tracking-wider">Quick-fill Example Scenarios</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {PRESETS.map((p, i) => (
            <button
              key={i}
              onClick={() => loadPreset(p)}
              className="text-left p-3 rounded-xl border-2 border-transparent hover:border-primary/40 bg-muted/50 hover:bg-muted transition-all group"
            >
              <div className="flex items-center gap-2 mb-1">
                <Badge className={catBadge[p.badge]?.color || 'bg-muted'}>{p.badge}</Badge>
                <span className="font-semibold text-sm">{p.label}</span>
              </div>
              <p className="text-xs text-muted-foreground">{p.desc}</p>
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Form */}
        <Card className="p-5 space-y-4">
          <h2 className="font-semibold text-lg">Sensor Readings</h2>

          <div className="space-y-1">
            <Label className="text-xs">Inverter ID</Label>
            <Input value={form.inverter_id} onChange={e => set('inverter_id', e.target.value)} className="h-9" />
          </div>

          <div className="grid grid-cols-2 gap-3">
            {FIELDS.map(f => (
              <div key={f.key} className="space-y-1">
                <Label className="text-xs">{f.label} <span className="text-muted-foreground">({f.unit})</span></Label>
                <Input
                  type="number" step="any"
                  placeholder={f.hint}
                  value={form[f.key]}
                  onChange={e => set(f.key, e.target.value)}
                  className="h-9"
                />
              </div>
            ))}
          </div>

          <div>
            <p className="text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">Optional Fields</p>
            <div className="grid grid-cols-2 gap-3">
              {OPT_FIELDS.map(f => (
                <div key={f.key} className="space-y-1">
                  <Label className="text-xs">{f.label}</Label>
                  <Input
                    type="number" step="any"
                    placeholder={f.hint}
                    value={form[f.key]}
                    onChange={e => set(f.key, e.target.value)}
                    className="h-9"
                  />
                </div>
              ))}
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 p-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
              <AlertTriangle className="h-4 w-4 flex-shrink-0" />
              {error}
            </div>
          )}

          <div className="flex gap-2 pt-1">
            <Button onClick={submit} disabled={loading} className="flex-1 gap-2">
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {loading ? 'Running Prediction…' : 'Run ML Prediction'}
            </Button>
            <Button variant="outline" onClick={reset} className="gap-2">
              <RotateCcw className="h-4 w-4" /> Reset
            </Button>
          </div>
        </Card>

        {/* Results */}
        <div className="space-y-4">
          {!pred && !loading && (
            <Card className="p-8 flex flex-col items-center justify-center text-center min-h-[300px]">
              <Zap className="h-12 w-12 text-muted-foreground/30 mb-3" />
              <h3 className="font-semibold text-lg">No Prediction Yet</h3>
              <p className="text-sm text-muted-foreground mt-1 max-w-sm">
                Fill in the sensor readings or select an example scenario above, then click <strong>Run ML Prediction</strong>.
              </p>
            </Card>
          )}

          {loading && (
            <Card className="p-8 flex flex-col items-center justify-center min-h-[300px]">
              <Loader2 className="h-10 w-10 text-primary animate-spin mb-3" />
              <p className="text-sm text-muted-foreground">Running XGBoost inference + SHAP analysis…</p>
            </Card>
          )}

          {pred && !loading && (
            <>
              {/* Input Verification */}
              <Card className="p-4 space-y-3">
                <h3 className="font-semibold text-sm">Input Verification</h3>
                <p className="text-xs text-muted-foreground">Confirming that the ML model received your exact input values.</p>
                <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
                  <span className="font-semibold text-muted-foreground">Field</span>
                  <span className="font-semibold text-muted-foreground">You Sent</span>
                  <span className="font-semibold text-muted-foreground">Model Received</span>
                  {FIELDS.map(f => {
                    const sent = (inputSent as any)[f.key];
                    const recv = inputEcho[f.key];
                    const match = sent !== undefined && recv !== undefined && Number(sent) === Number(recv);
                    return [
                      <span key={f.key + '-l'} className="text-muted-foreground">{f.label}</span>,
                      <span key={f.key + '-s'} className="font-mono">{sent ?? '—'} {f.unit}</span>,
                      <span key={f.key + '-r'} className={`font-mono ${match ? 'text-emerald-600' : recv !== undefined ? 'text-amber-600' : 'text-muted-foreground'}`}>
                        {recv != null ? `${recv} ${f.unit}` : '—'} {match ? '\u2713' : ''}
                      </span>,
                    ];
                  })}
                </div>
              </Card>

              {/* Category + Confidence */}
              <Card className="p-5 space-y-4">
                <div className="flex items-center gap-3">
                  <div className={`h-12 w-12 rounded-xl flex items-center justify-center ${catInfo?.color || 'bg-muted'}`}>
                    <CatIcon className="h-6 w-6" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-bold text-xl">Category {cat}</h3>
                    <p className="text-sm text-muted-foreground">{catInfo?.label}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-extrabold">{(pred.confidence * 100).toFixed(1)}%</p>
                    <p className="text-xs text-muted-foreground">Confidence</p>
                  </div>
                </div>

                <Progress value={pred.confidence * 100} className="h-2" />

                {pred.fault && (
                  <div className="flex items-start gap-2 p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                    <AlertTriangle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-semibold text-destructive">Fault Detected</p>
                      <p className="text-sm text-destructive/80">{pred.fault}</p>
                    </div>
                  </div>
                )}

                {!pred.fault && (
                  <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                    <p className="text-sm text-emerald-700 font-medium">No fault detected — inverter operating normally</p>
                  </div>
                )}
              </Card>

              {/* Class Probabilities */}
              <Card className="p-5 space-y-3">
                <h3 className="font-semibold">Class Probabilities</h3>
                <div className="space-y-2">
                  {probData.map(p => (
                    <div key={p.name} className="flex items-center gap-3">
                      <span className="text-xs w-28 text-muted-foreground capitalize">{p.name}</span>
                      <div className="flex-1 h-5 bg-muted rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${
                            p.name.includes('shutdown') ? 'bg-red-500' :
                            p.name.includes('degradation') ? 'bg-orange-500' : 'bg-emerald-500'
                          }`}
                          style={{ width: `${p.value}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono font-bold w-16 text-right">{p.value}%</span>
                    </div>
                  ))}
                </div>
              </Card>

              {/* SHAP Features */}
              {shapData.length > 0 && (
                <Card className="p-5 space-y-3">
                  <h3 className="font-semibold">SHAP Feature Importance (Top 5)</h3>
                  <p className="text-xs text-muted-foreground">
                    These are <strong>SHAP contribution values</strong> — they show how much each internal model feature pushed the prediction toward risk (red) or safety (blue).
                    They are NOT the raw sensor values you entered. The model internally derives 183 engineered features from your 6 inputs.
                  </p>
                  <div className="h-52">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={shapData} layout="vertical" margin={{ left: 120, right: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis type="number" tick={{ fontSize: 11 }} />
                        <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={120} />
                        <Tooltip formatter={(v: any) => Number(v).toFixed(6)} />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                          {shapData.map((entry, i) => (
                            <Cell key={i} fill={entry.value >= 0 ? 'hsl(0, 84%, 60%)' : 'hsl(217, 91%, 60%)'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              )}

              {/* Raw JSON */}
              <Card className="p-4">
                <details>
                  <summary className="text-xs font-semibold text-muted-foreground cursor-pointer hover:text-foreground">
                    View Raw ML Response JSON
                  </summary>
                  <pre className="text-xs bg-muted p-3 rounded-lg mt-2 overflow-x-auto max-h-60">
                    {JSON.stringify(cleanPred, null, 2)}
                  </pre>
                </details>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
