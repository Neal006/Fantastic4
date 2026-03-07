import { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Send, Bot, User, AlertTriangle, Zap, FileDown, Info,
  CheckCircle2, XCircle, Loader2, ChevronRight,
} from 'lucide-react';
import { chatbotApi } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';

interface Message {
  role: 'user' | 'bot';
  content: string;
  sources?: string[];
}

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

function renderMessage(content: string) {
  // Simple bold markdown rendering **text** → <strong>text</strong>
  const parts = content.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={i}>{part.slice(2, -2)}</strong>
      : <span key={i}>{part}</span>
  );
}

export default function ChatbotPage() {
  const location = useLocation();
  const context = (location.state as any)?.context as any | undefined;
  const inverterId = context?.inverter_id;  // DB UUID (not used for GenAI)
  const inverterName = context?.inverter_name || inverterId;  // e.g. INV-P1-L2-0

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'bot',
      content: inverterName
        ? `I'm SolarGuard AI. I can see you want to discuss **${inverterName}**. Ask me anything, or use the quick actions below.`
        : `I'm SolarGuard AI. I can help you understand inverter faults, SHAP analysis, and maintenance procedures. Ask me about any inverter or issue!`,
    },
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanation, setExplanation] = useState<any | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Health check
  const { data: health, isError: healthError } = useQuery({
    queryKey: ['genai-health'],
    queryFn: chatbotApi.getHealth,
    retry: 1,
    staleTime: 30_000,
    refetchInterval: 60_000,
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const send = async (overrideMsg?: string) => {
    const userMsg = overrideMsg ?? input.trim();
    if (!userMsg || typing) return;
    if (!overrideMsg) setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setTyping(true);

    try {
      const result = await chatbotApi.query(userMsg, sessionId);
      if (!sessionId && result.session_id) setSessionId(result.session_id);
      setMessages(prev => [
        ...prev,
        { role: 'bot', content: result.response, sources: result.sources_used },
      ]);
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'bot', content: 'The AI assistant is currently unavailable. Please try again.' },
      ]);
    } finally {
      setTyping(false);
    }
  };

  const fetchExplanation = async () => {
    if (!inverterName || explanationLoading) return;
    setExplanationLoading(true);
    setExplanation(null);
    try {
      const data = await chatbotApi.getExplanation(inverterName);
      setExplanation(data);
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'bot', content: `❌ Could not fetch AI explanation for **${inverterName}**. Make sure the GenAI server is running.` },
      ]);
    } finally {
      setExplanationLoading(false);
    }
  };

  const downloadPdf = async () => {
    if (!inverterName || pdfLoading) return;
    setPdfLoading(true);
    try {
      // Trigger ticket generation first (ensures PDF exists)
      await chatbotApi.generateTicket(inverterName);
      const token = sessionStorage.getItem('sw_token');
      const pdfUrl = chatbotApi.getPdfUrl(inverterName);
      // Open PDF URL with auth header via fetch → blob download
      const res = await fetch(pdfUrl, {
        headers: { Authorization: `Bearer ${token}` },
        credentials: 'include',
      });
      if (!res.ok) throw new Error('PDF unavailable');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${inverterName}-ticket.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      setMessages(prev => [
        ...prev,
        { role: 'bot', content: `❌ Could not download PDF ticket for **${inverterName}**. The GenAI server may be unavailable.` },
      ]);
    } finally {
      setPdfLoading(false);
    }
  };

  const riskPercent = explanation ? Math.round(explanation.risk_score * 100) : null;

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)] gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold">AI Chatbot</h1>
          {inverterName && <Badge variant="secondary">Context: {inverterName}</Badge>}
        </div>
        {/* GenAI Status */}
        <div className="flex items-center gap-1.5 text-xs">
          {healthError ? (
            <span className="flex items-center gap-1 text-destructive">
              <XCircle className="h-3.5 w-3.5" /> GenAI Offline
            </span>
          ) : health ? (
            <span className="flex items-center gap-1 text-green-600">
              <CheckCircle2 className="h-3.5 w-3.5" />
              GenAI Online · {health.inverters_monitored} inverters
            </span>
          ) : (
            <span className="flex items-center gap-1 text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> Checking…
            </span>
          )}
        </div>
      </div>

      {/* GenAI Offline banner */}
      {healthError && (
        <div className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
          <AlertTriangle className="h-4 w-4 flex-shrink-0" />
          <span>The GenAI AI engine is offline. Start the Python server (<code>uvicorn app.main:app --port 8000</code>) to enable AI features.</span>
        </div>
      )}

      <div className="flex flex-1 gap-3 overflow-hidden">
        {/* Chat panel */}
        <Card className="flex-1 flex flex-col rounded-xl overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.map((msg, i) => (
              <div key={i} className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'bot' && (
                  <div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <Bot className="h-4 w-4 text-primary" />
                  </div>
                )}
                <div className={`max-w-[80%] space-y-1`}>
                  <div className={`rounded-xl px-3.5 py-2.5 text-sm whitespace-pre-wrap ${msg.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'
                    }`}>
                    {renderMessage(msg.content)}
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <p className="text-[10px] text-muted-foreground pl-1">
                      Sources: {msg.sources.join(', ')}
                    </p>
                  )}
                </div>
                {msg.role === 'user' && (
                  <div className="h-7 w-7 rounded-full bg-primary flex items-center justify-center flex-shrink-0 mt-0.5">
                    <User className="h-4 w-4 text-primary-foreground" />
                  </div>
                )}
              </div>
            ))}
            {typing && (
              <div className="flex gap-2">
                <div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
                <div className="bg-muted rounded-xl px-4 py-3">
                  <div className="flex gap-1">
                    <span className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="h-2 w-2 rounded-full bg-muted-foreground/40 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="border-t p-3 space-y-2">
            {/* Quick actions when inverter context exists */}
            {inverterName && (
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1"
                  onClick={() => send(`Explain the risk for inverter ${inverterName}`)}
                  disabled={typing || !!healthError}
                >
                  <Zap className="h-3 w-3" /> Ask about {inverterName}
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-xs h-7 gap-1"
                  onClick={() => send(`What maintenance steps should I follow for ${inverterName}?`)}
                  disabled={typing || !!healthError}
                >
                  <Info className="h-3 w-3" /> Maintenance steps
                </Button>
              </div>
            )}
            <div className="flex gap-2">
              <Input
                placeholder="Ask about an inverter fault, SHAP analysis, maintenance..."
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && send()}
                className="flex-1"
                disabled={typing || !!healthError}
              />
              <Button size="icon" onClick={() => send()} disabled={!input.trim() || typing || !!healthError}>
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </Card>

        {/* AI Explanation panel — only when inverter context exists */}
        {inverterName && (
          <div className="w-80 flex flex-col gap-3">
            <Card className="p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="font-semibold text-sm">AI Risk Explanation</h2>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 text-xs gap-1"
                  onClick={fetchExplanation}
                  disabled={explanationLoading || !!healthError}
                >
                  {explanationLoading
                    ? <Loader2 className="h-3 w-3 animate-spin" />
                    : <Zap className="h-3 w-3" />}
                  {explanation ? 'Refresh' : 'Explain'}
                </Button>
              </div>

              {explanationLoading && (
                <div className="space-y-2">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-16 w-full" />
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-4 w-full" />
                  <Skeleton className="h-4 w-full" />
                </div>
              )}

              {explanation && !explanationLoading && (
                <div className="space-y-3 text-sm">
                  {/* Risk badge + urgency */}
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-bold text-lg">{riskPercent}% risk</span>
                    <Badge className={URGENCY_COLORS[explanation.urgency] ?? 'bg-muted'}>
                      {URGENCY_LABEL[explanation.urgency] ?? explanation.urgency.toUpperCase()}
                    </Badge>
                  </div>

                  {/* Summary */}
                  <p className="text-muted-foreground leading-relaxed">{explanation.summary}</p>

                  {/* Key factors */}
                  {explanation.key_factors?.length > 0 && (
                    <div>
                      <p className="font-medium mb-1">Key Risk Factors</p>
                      <ul className="space-y-1">
                        {explanation.key_factors.map((f: any, i: number) => (
                          <li key={i} className="flex items-start gap-1.5">
                            <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 mt-0.5 text-muted-foreground" />
                            <span>
                              <span className="font-medium">{f.feature}</span>
                              {f.raw_value ? ` (${f.raw_value})` : ''} — {f.impact}
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Recommended actions */}
                  {explanation.recommended_actions?.length > 0 && (
                    <div>
                      <p className="font-medium mb-1">Recommended Actions</p>
                      <ol className="space-y-1 list-decimal list-inside text-muted-foreground">
                        {explanation.recommended_actions.map((a: string, i: number) => (
                          <li key={i}>{a}</li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {/* Disclaimer */}
                  {explanation.disclaimer && (
                    <p className="text-[10px] text-muted-foreground italic border-t pt-2">{explanation.disclaimer}</p>
                  )}
                </div>
              )}

              {!explanation && !explanationLoading && (
                <p className="text-sm text-muted-foreground">
                  Click <strong>Explain</strong> to get an AI-generated risk analysis for <strong>{inverterName}</strong>.
                </p>
              )}
            </Card>

            {/* PDF Ticket Download */}
            <Card className="p-4">
              <h2 className="font-semibold text-sm mb-2">Maintenance Ticket</h2>
              <p className="text-xs text-muted-foreground mb-3">
                Generate a professional PDF maintenance ticket for this inverter.
              </p>
              <Button
                variant="default"
                size="sm"
                className="w-full gap-2"
                onClick={downloadPdf}
                disabled={pdfLoading || !!healthError}
              >
                {pdfLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileDown className="h-4 w-4" />}
                {pdfLoading ? 'Generating…' : 'Download PDF Ticket'}
              </Button>
            </Card>

            <Card className="p-4">
              <h2 className="font-semibold text-sm mb-2">Reference Report</h2>
              <p className="text-xs text-muted-foreground mb-3">
                Download the SolarWatch analysis report with methodology, model details, and comparative results.
              </p>
              <Button
                variant="outline"
                size="sm"
                className="w-full gap-2"
                onClick={() => {
                  const a = document.createElement('a');
                  a.href = chatbotApi.getReferencePdfUrl();
                  a.download = 'SolarWatch-Analysis-Report.pdf';
                  a.click();
                }}
              >
                <FileDown className="h-4 w-4" />
                Download Analysis Report
              </Button>
            </Card>
          </div>
        )}

        {/* Reference PDF — always visible (not just with inverter context) */}
        {!inverterName && (
          <div className="w-80 flex flex-col gap-3">
            <Card className="p-4">
              <h2 className="font-semibold text-sm mb-2">Reference Report</h2>
              <p className="text-xs text-muted-foreground mb-3">
                Download the SolarWatch analysis report with methodology, model details, and comparative results.
              </p>
              <Button
                variant="default"
                size="sm"
                className="w-full gap-2"
                onClick={() => {
                  const token = sessionStorage.getItem('sw_token');
                  const a = document.createElement('a');
                  a.href = chatbotApi.getReferencePdfUrl();
                  a.download = 'SolarWatch-Analysis-Report.pdf';
                  a.click();
                }}
              >
                <FileDown className="h-4 w-4" />
                Download Analysis Report
              </Button>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
