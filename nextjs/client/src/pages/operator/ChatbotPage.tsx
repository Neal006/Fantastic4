import { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Send, Bot, User } from 'lucide-react';
import { chatbotApi } from '@/lib/api';

interface Message {
  role: 'user' | 'bot';
  content: string;
}

const DEFAULT_MSG = "I'm SolarWatch AI. I can help you understand inverter faults, SHAP analysis, and maintenance procedures. Ask me about any inverter or issue!";

export default function ChatbotPage() {
  const location = useLocation();
  const context = (location.state as any)?.context as any | undefined;
  const inverterId = context?.inverter_id;

  const [messages, setMessages] = useState<Message[]>([
    { role: 'bot', content: inverterId ? `I see you want to discuss **${inverterId}**. What would you like to know?` : DEFAULT_MSG }
  ]);
  const [input, setInput] = useState('');
  const [typing, setTyping] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  const send = async () => {
    if (!input.trim() || typing) return;
    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setTyping(true);

    try {
      const conversationHistory = messages.map(m => ({ role: m.role === 'bot' ? 'assistant' : 'user', content: m.content }));
      const result = await chatbotApi.query(userMsg, context, conversationHistory);
      setMessages(prev => [...prev, { role: 'bot', content: result.response }]);
    } catch {
      setMessages(prev => [...prev, { role: 'bot', content: 'The AI assistant is currently unavailable. Please try again.' }]);
    } finally {
      setTyping(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      <div className="flex items-center gap-2 mb-4">
        <h1 className="text-2xl font-bold">AI Chatbot</h1>
        {inverterId && <Badge variant="secondary">Context: {inverterId}</Badge>}
      </div>

      <Card className="flex-1 flex flex-col rounded-xl overflow-hidden">
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-2 ${msg.role === 'user' ? 'justify-end' : ''}`}>
              {msg.role === 'bot' && (
                <div className="h-7 w-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
              )}
              <div className={`max-w-[80%] rounded-xl px-3.5 py-2.5 text-sm whitespace-pre-wrap ${msg.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-muted'
                }`}>
                {msg.content}
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

        <div className="border-t p-3 flex gap-2">
          <Input
            placeholder="Ask about an inverter fault, SHAP analysis, maintenance..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && send()}
            className="flex-1"
            disabled={typing}
          />
          <Button size="icon" onClick={send} disabled={!input.trim() || typing}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </Card>
    </div>
  );
}
