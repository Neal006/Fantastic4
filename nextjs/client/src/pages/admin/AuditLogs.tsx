import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Loader2, Search } from 'lucide-react';

export default function AuditLogs() {
  const [search, setSearch] = useState('');

  const { data: logs = [], isLoading } = useQuery({
    queryKey: ['audit-logs'],
    queryFn: () => adminApi.getAuditLogs({ limit: '100' }),
    refetchInterval: 30000,
  });

  const filtered = (logs as any[]).filter((log: any) =>
    log.action?.toLowerCase().includes(search.toLowerCase()) ||
    log.user_name?.toLowerCase().includes(search.toLowerCase())
  );

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Audit Logs</h1>

      <div className="relative w-64">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        <Input placeholder="Search logs..." value={search} onChange={e => setSearch(e.target.value)} className="pl-8 h-8 text-xs" />
      </div>

      <div className="rounded-xl border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Time</TableHead>
              <TableHead>User</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Action</TableHead>
              <TableHead>Details</TableHead>
              <TableHead>IP</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.length === 0 ? (
              <TableRow><TableCell colSpan={6} className="text-center py-8 text-muted-foreground">No logs found</TableCell></TableRow>
            ) : filtered.map((log: any) => (
              <TableRow key={log.id} className="hover:bg-muted/50">
                <TableCell className="text-xs">{new Date(log.created_at).toLocaleString()}</TableCell>
                <TableCell className="text-sm font-medium">{log.user_name || log.user_id}</TableCell>
                <TableCell>
                  <Badge variant={log.user_role === 'admin' ? 'default' : 'secondary'} className="text-xs capitalize">{log.user_role}</Badge>
                </TableCell>
                <TableCell className="text-sm font-mono">{log.action}</TableCell>
                <TableCell className="text-xs text-muted-foreground max-w-[200px] truncate">
                  {typeof log.details === 'object' ? Object.values(log.details || {}).join(', ') : log.details}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">{log.ip_address}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
