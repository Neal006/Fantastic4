import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { adminApi } from '@/lib/api';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2 } from 'lucide-react';
import { toast } from 'sonner';

export default function AdminAlerts() {
  const [filterType, setFilterType] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const qc = useQueryClient();

  const { data: alerts = [], isLoading } = useQuery({
    queryKey: ['admin-alerts', filterType, filterStatus],
    queryFn: () => {
      const params: Record<string, string> = {};
      if (filterType !== 'all') params.type = filterType;
      if (filterStatus !== 'all') params.acknowledged = filterStatus === 'acknowledged' ? 'true' : 'false';
      return adminApi.getAlerts(params);
    },
    refetchInterval: 20000,
  });

  if (isLoading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );

  const sevBadge = (type: string) => {
    if (type === 'critical') return <Badge className="bg-sw-critical text-primary-foreground text-xs">Critical</Badge>;
    if (type === 'warning') return <Badge className="bg-sw-warning text-primary-foreground text-xs">Warning</Badge>;
    return <Badge variant="secondary" className="text-xs">Info</Badge>;
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Alert Overview</h1>

      <div className="flex flex-wrap gap-2">
        <Select value={filterType} onValueChange={setFilterType}>
          <SelectTrigger className="w-36 h-8 text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
            <SelectItem value="warning">Warning</SelectItem>
            <SelectItem value="info">Info</SelectItem>
          </SelectContent>
        </Select>
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="w-40 h-8 text-xs"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="pending">Unacknowledged</SelectItem>
            <SelectItem value="acknowledged">Acknowledged</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="rounded-xl border overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Time</TableHead>
              <TableHead>Inverter</TableHead>
              <TableHead>Plant</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Message</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Acknowledged By</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {(alerts as any[]).length === 0 ? (
              <TableRow><TableCell colSpan={7} className="text-center py-8 text-muted-foreground">No alerts found</TableCell></TableRow>
            ) : (alerts as any[]).map((alert: any) => (
              <TableRow key={alert.id} className="hover:bg-muted/50">
                <TableCell className="text-xs">{new Date(alert.created_at).toLocaleString()}</TableCell>
                <TableCell className="font-mono text-sm">{alert.inverter_name}</TableCell>
                <TableCell className="text-xs">{alert.plant_name} › {alert.block_name}</TableCell>
                <TableCell>{sevBadge(alert.type)}</TableCell>
                <TableCell className="text-sm max-w-[180px] truncate">{alert.message}</TableCell>
                <TableCell>
                  <Badge variant={alert.acknowledged ? 'secondary' : 'destructive'} className="text-xs">
                    {alert.acknowledged ? 'Acknowledged' : 'Pending'}
                  </Badge>
                </TableCell>
                <TableCell className="text-xs text-muted-foreground">
                  {alert.acknowledged_by_name || '—'}
                  {alert.acknowledged_at ? ` at ${new Date(alert.acknowledged_at).toLocaleTimeString()}` : ''}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
